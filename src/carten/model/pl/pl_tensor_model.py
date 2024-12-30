"""Lightning model for predicting tensorial properties of materials and molecules.

This is supposed to be used with the `carten.model.tensor.StructureTensorModel`
and `carten.model.tensor.AtomicTensorModel` models.
"""

from typing import Any

import torch
from ema_pytorch import EMA
from lightning import LightningModule
from lightning.pytorch.cli import instantiate_class
from lightning.pytorch.utilities import grad_norm
from lightning.pytorch.utilities.types import STEP_OUTPUT
from line_profiler import profile
from torch import nn
from torchmetrics import MeanAbsoluteError, MeanSquaredError

from carten.data.utils import get_edge_vec


class BaseLitModule(LightningModule):
    """
    Base lightning model for predicting tensorial properties of materials and molecules.

    This should not be used directly, but should be subclassed by the specific models.
    """

    def __init__(
        self,
        model: nn.Module,
        loss_hparams: dict[str, Any] = None,
        metrics_hparams: dict[str, float] = None,
        optimizer_hparams: dict[str, Any] = None,
        lr_scheduler_hparams: dict[str, Any] = None,
        ema_hparams: dict[str, Any] = None,
        other_hparams: dict[str, Any] = None,
    ):
        super().__init__()

        self.save_hyperparameters(ignore=["model"])

        self.model = model
        self.loss_hparams = loss_hparams
        self.metrics_hparams = metrics_hparams
        self.optimizer_hparams = optimizer_hparams
        self.lr_scheduler_hparams = lr_scheduler_hparams
        self.ema_hparams = ema_hparams
        self.ema = EMA(self.model, **self.ema_hparams)

        self.metrics_type = self.metrics_hparams.get("type", None)
        if self.metrics_type == "mae":
            MetricClass = MeanAbsoluteError
        elif self.metrics_type == "mse":
            MetricClass = MeanSquaredError
        else:
            raise ValueError(f"Unknown metrics type: {self.metrics_type}")

        # validation will only start after this
        self.validation_start_epoch = self.metrics_hparams.get(
            "validation_start_epoch", 0
        )

        self.metrics = nn.ModuleDict(
            {
                f"metrics_{mode}_{rank}": MetricClass()
                for mode in ["train", "val", "test", "val_ema", "test_ema"]
                for rank in self.loss_hparams["target_signature"]
            }
        )

    @profile
    def forward(self, batch):
        """Compute model output."""
        # Note, it is tempting to compute the edge_vector in the collate_fn of the
        # dataloader. However, this will not work will pytorch lightning, internally
        # it does something to the batch, modifying both pos and edge_index. As a
        # result, the edge_vector is not directly derived from pos, and thus we won't
        # be able to compute the forces.
        # This is why we compute the edge_vector here.
        #
        # TODO, the above comments only applies to interatomic potentials. For
        #  atomic/structure tensor models, we can compute the edge_vector in the
        #  collate_fn.

        return self.model(
            edge_vector=self._get_edge_vector(batch),
            edge_idx=batch.edge_index,
            atom_type=batch.atom_type,
            num_atoms=batch.num_atoms,
        )

    @profile
    def forward_ema(self, batch):
        """Same as `forward, but use the EMA model instead of the original model."""

        return self.ema(
            edge_vector=self._get_edge_vector(batch),
            edge_idx=batch.edge_index,
            atom_type=batch.atom_type,
            num_atoms=batch.num_atoms,
        )

    def training_step(self, batch, batch_idx):
        ref = batch.y[self.loss_hparams["target_name"]]
        batch_size = batch.num_graphs

        pred = self(batch)
        losses = self.compute_loss(pred, ref)
        metrics = self.compute_metrics(pred, ref, "train")
        d = {**losses, **metrics}

        self.log_dict(
            d, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size
        )

        return losses["train/loss_total"]

    def validation_step(self, batch, batch_idx):
        self._val_test_step(
            batch, batch_idx, mode="val", start_epoch=self.validation_start_epoch
        )

    def test_step(self, batch, batch_idx):
        self._val_test_step(batch, batch_idx, mode="test")

    def _val_test_step(self, batch, batch_idx, mode: str, start_epoch: int = 0):
        ref = batch.y[self.loss_hparams["target_name"]]
        batch_size = batch.num_graphs

        # use current model
        if self.current_epoch >= start_epoch:
            pred = self(batch)
        else:
            # TODO, this needs to be updated as a dict
            # dummy values to skip validation for the first few epochs
            pred = torch.zeros(1, dtype=ref.dtype, device=ref.device)
            ref = 1e10 * torch.ones(1, dtype=ref.dtype, device=ref.device)

        metrics = self.compute_metrics(pred, ref, mode)
        self.log_dict(
            metrics,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch_size,
        )

        # use EMA model
        if self.current_epoch >= start_epoch:
            pred = self.forward_ema(batch)
        else:
            # TODO, this needs to be updated as a dict
            # dummy values to skip validation for the first few epochs
            pred = torch.zeros(1, dtype=ref.dtype, device=ref.device)
            ref = 1e10 * torch.ones(1, dtype=ref.dtype, device=ref.device)

        metrics = self.compute_metrics(pred, ref, mode + "_ema")
        self.log_dict(
            metrics,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch_size,
        )

    def compute_loss(self, pred: dict, ref: dict):
        """
        Total loss:
            loss_total = loss
        """
        raise NotImplementedError("Subclass must implement this method.")

    def compute_metrics(self, pred: dict, ref: dict, mode: str):
        """
        MAE:
            MAE = mean_k |E_pred - E_ref|
        """
        raise NotImplementedError("Subclass must implement this method.")

    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)
        self.ema.update()

    def configure_optimizers(self):
        # optimizer
        # use self.model.parameters() instead of self.parameters() because the latter
        # also includes the ema model parameters
        model_params = self.model.parameters()
        optimizer = instantiate_class(model_params, self.optimizer_hparams)

        # lr scheduler
        class_path = self.lr_scheduler_hparams.get("class_path")
        if class_path is None or class_path == "none":
            scheduler = None
            monitor = None
        else:
            if class_path == "torch.optim.lr_scheduler.ReduceLROnPlateau":
                monitor = self.lr_scheduler_hparams.get("monitor")
                if monitor is None:
                    raise ValueError(
                        "Please provide a `monitor` for the learning rate scheduler: "
                        f"ReduceLROnPlateau."
                    )
            else:
                monitor = None
            scheduler = instantiate_class(optimizer, self.lr_scheduler_hparams)

        if scheduler is None:
            return optimizer
        else:
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "monitor": monitor},
            }

    @staticmethod
    def _get_edge_vector(batch):
        try:
            cell = batch.cell
            shift_vec = batch.shift_vec
        except AttributeError:
            # this happens for molecules, no pbc needed, and no cell
            cell = None
            shift_vec = None

        edge_vector = get_edge_vec(
            batch.pos, shift_vec, cell, batch.edge_index, batch.batch
        )

        return edge_vector

    ## TODO, for DEBUG only, should be commented out
    # Grad norm computation should be called after `configure_gradient_clipping()` in
    # case we are clipping the gradients and wand to show the clipped gradients.
    # For the fit loop hook calling order,
    # see https://lightning.ai/docs/pytorch/2.2.1/common/lightning_module.html#hooks
    #
    # You might need to increase max_epoch for this to be executed and show up in wandb
    # Compute the 2-norm for each layer
    # def on_train_batch_end(self, outputs: STEP_OUTPUT, batch: Any, batch_idx: int):
    #     norms = grad_norm(self.model, norm_type=2)
    #     self.log_dict(norms, prog_bar=False)


# class AtomicTensorLitModule(LightningModule):
#     def compute_loss(self, pred, ref):
#         """
#         Total loss:
#             loss_total = loss
#         """
#         # self.loss_hparams['energy_ratio']
#         loss = nn.functional.mse_loss(pred, ref, reduction="mean")
#         losses = {"train/loss": loss}
#
#         return losses
#
#     def compute_metrics(self, pred: dict, ref: dict, mode: str):
#         """
#         MAE:
#             MAE = mean_k |E_pred - E_ref|
#         """
#         self.metrics[f"metrics_{mode}"](pred, ref)
#         metrics = {f"{mode}/{self.metrics_type}": self.metrics[f"metrics_{mode}"]}
#
#         return metrics


class StructureTensorLitModule(BaseLitModule):
    def compute_loss(self, pred: dict, ref: dict):
        """
        Weighted MSE loss.
        """
        losses = {}
        for rank in pred:
            p = pred[rank]
            r = ref[str(rank)]

            ratio = self.loss_hparams["ratio"][rank]
            losses[f"train/loss_rank-{rank}"] = ratio * nn.functional.mse_loss(
                p, r, reduction="mean"
            )

        losses["train/loss_total"] = sum(losses.values())

        return losses

    def compute_metrics(self, pred: dict, ref: dict, mode: str):
        """
        MAE:
            MAE = mean_k |E_pred - E_ref|
        """
        metrics = {}
        for rank in pred:
            p = pred[rank]
            r = ref[str(rank)]

            # call the metric object
            name = f"metrics_{mode}_{rank}"
            self.metrics[name](p, r)

            # record the metric object to return
            metrics[f"{mode}/{self.metrics_type}_{rank}"] = self.metrics[name]

        return metrics
