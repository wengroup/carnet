"""Lightning model for predicting tensorial properties of materials and molecules.

This is supposed to be used with the `carnet.model.tensor.StructureTensorModel`
and `carnet.model.tensor.AtomicTensorModel` models.
"""

from typing import Any

import torch
from ema_pytorch import EMA
from lightning import LightningModule
from lightning.pytorch.cli import instantiate_class
from torch import Tensor, nn
from torchmetrics import MeanAbsoluteError, MeanSquaredError

from carnet.core.convert import Converter
from carnet.model.elastic import get_voigt_projection_tensor


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

        loss_type = self.loss_hparams.get("type", None)
        if loss_type == "mse":
            self.loss_func = nn.functional.mse_loss
        elif loss_type == "mae":
            self.loss_func = nn.functional.l1_loss
        elif loss_type == "huber":
            delta = self.loss_hparams.get("delta", None)
            if delta is None:
                raise ValueError(
                    "Please provide `delta` in loss params for huber loss."
                )
            self.loss_func = nn.HuberLoss(delta=delta, reduction="mean")
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")

        self.metrics_type = self.metrics_hparams.get("type", None)
        if self.metrics_type == "mae":
            MetricClass = MeanAbsoluteError
        elif self.metrics_type == "mse":
            MetricClass = MeanSquaredError
        else:
            raise ValueError(f"Unknown metrics type: {self.metrics_type}")

        # metrics for natural tensors
        self.metrics = nn.ModuleDict(
            {
                f"metrics_{mode}_{rank}": MetricClass()
                for mode in ["train", "val", "test", "val_ema", "test_ema"]
                for rank in self.loss_hparams["target_signature"]
            }
        )

        #
        # The below is only needed for cartesian tensors
        #

        # metrics for cartesian tensors
        self.target_mode = self.loss_hparams.get("target_mode", "natural")

        if self.target_mode == "natural":
            self.register_buffer("metrics_cartesian", None)
            self.register_buffer("converter", None)
        elif self.target_mode in ["full", "voigt"]:
            self.metrics_cartesian = nn.ModuleDict(
                {
                    f"metrics_{mode}_cartesian": MetricClass()
                    for mode in ["train", "val", "test", "val_ema", "test_ema"]
                }
            )

            # values needed to convert natural tensors to cartesian tensors
            symmetry = self.loss_hparams.get("target_symmetry")
            if symmetry is None:
                raise ValueError('"target_symmetry" must be provided in loss_hparams.')
            self.converter = Converter(symmetry)

        else:
            raise ValueError(f"Unknown target mode: {self.target_mode}")

        # values needed to convert cartesian tensors to Voigt tensors
        if self.target_mode == "voigt":
            # TODO, this is hard coded for elastic tensor
            # C_ab = M_ijkl^ab C_ijkl
            M = get_voigt_projection_tensor()

            # Reshape it to M_xy, where x represents ijkl and y represents ab,
            # so we can do multiplication
            M = M.reshape(-1, 6 * 6)

            self.register_buffer("M", M)
        else:
            self.register_buffer("M", None)

    def forward(self, batch):
        """Compute model output."""

        # Atomic selector only needed for atomic tensor model, not for structure tensor
        # model.
        atomic_selector = (
            None if "atomic_selector" not in batch.y else batch.y["atomic_selector"]
        )

        return self.model(
            edge_vector=batch.edge_vector,
            edge_idx=batch.edge_index,
            atom_type=batch.atom_type,
            num_atoms=batch.num_atoms,
            atomic_selector=atomic_selector,
            batch=batch.batch,
        )

    def forward_ema(self, batch):
        """Same as `forward, but use the EMA model instead of the original model."""

        atomic_selector = (
            None if "atomic_selector" not in batch.y else batch.y["atomic_selector"]
        )
        return self.ema(
            edge_vector=batch.edge_vector,
            edge_idx=batch.edge_index,
            atom_type=batch.atom_type,
            num_atoms=batch.num_atoms,
            atomic_selector=atomic_selector,
            batch=batch.batch,
        )

    def training_step(self, batch, batch_idx):
        batch_size = batch.num_graphs

        ref_nat = batch.y[self.loss_hparams["target_name"] + "_natural"]
        pred_nat = self(batch)

        # Get predictions
        if self.target_mode == "natural":
            losses = self.compute_loss_nat(pred_nat, ref_nat)
            metrics = self.compute_metrics(pred_nat, ref_nat, mode="train")

        elif self.target_mode in ["full", "voigt"]:
            pred = self.to_cartesian(pred_nat)
            ref = batch.y[
                self.loss_hparams["target_name"]
                + "_"
                + self.loss_hparams["target_mode"]
            ]

            losses = self.compute_loss_cart(pred, ref)
            metrics = self.compute_metrics(pred_nat, ref_nat, pred, ref, mode="train")
        else:
            raise ValueError(f"Unknown target mode: {self.target_mode}")

        d = {**losses, **metrics}

        self.log_dict(
            d, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size
        )

        return losses["train/loss_total"]

    def validation_step(self, batch, batch_idx):
        self._val_test_step(batch, batch_idx, mode="val")

    def test_step(self, batch, batch_idx):
        self._val_test_step(batch, batch_idx, mode="test")

    def _val_test_step(self, batch, batch_idx, mode: str):
        batch_size = batch.num_graphs

        ref_nat = batch.y[self.loss_hparams["target_name"] + "_natural"]
        ref = batch.y[
            self.loss_hparams["target_name"] + "_" + self.loss_hparams["target_mode"]
        ]

        # use current model
        pred_nat = self(batch)

        if self.target_mode in ["full", "voigt"]:
            pred = self.to_cartesian(pred_nat)
        else:
            pred = None

        metrics = self.compute_metrics(pred_nat, ref_nat, pred, ref, mode)

        # use EMA model
        pred_nat = self.forward_ema(batch)
        if self.target_mode in ["full", "voigt"]:
            pred = self.to_cartesian(pred_nat)
        else:
            pred = None

        metrics_eam = self.compute_metrics(pred_nat, ref_nat, pred, ref, mode + "_ema")
        metrics.update(metrics_eam)

        self.log_dict(
            metrics,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch_size,
        )

    def compute_loss_nat(self, pred: dict, ref: dict):
        """
        Total loss:
            loss_total = loss
        """
        raise NotImplementedError("Subclass must implement this method.")

    def compute_loss_cart(self, pred: Tensor, ref: Tensor):
        raise NotImplementedError("Subclass must implement this method.")

    def compute_metrics(
        self,
        pred_nat: dict,
        ref_nat: dict,
        pred_cart: Tensor = None,
        ref_cart: Tensor = None,
        mode: str = "train",
    ):

        metrics = self.compute_metrics_nat(pred_nat, ref_nat, mode)
        if self.target_mode in ["full", "voigt"]:
            metrics_cart = self.compute_metrics_cart(pred_cart, ref_cart, mode)
            metrics.update(metrics_cart)

        return metrics

    def compute_metrics_nat(self, pred: dict, ref: dict, mode: str):
        """
        MAE:
            MAE = mean_k |E_pred - E_ref|
        """
        raise NotImplementedError("Subclass must implement this method.")

    def compute_metrics_cart(self, pred: Tensor, ref: Tensor, mode: str):
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

    def to_cartesian(self, pred: dict):
        """
        Convert the predicted tensors to Cartesian tensors.
        """

        # Convert to Voigt tensor if needed
        if self.M is not None:
            # Convert natural tensors to Cartesian tensors (n_configs, 3**rank)
            pred = self.converter.to_ordinary_tensor(pred, flatten_tensor_dim=True)
            # shape (n_configs, 6*6)
            pred = torch.matmul(pred, self.M)
        else:
            # Convert natural tensors to Cartesian tensors (n_configs, 3,3,...,3)
            pred = self.converter.to_ordinary_tensor(pred, flatten_tensor_dim=False)

        return pred


class StructureTensorLitModule(BaseLitModule):
    def compute_loss_nat(self, pred: dict, ref: dict):
        """
        Weighted MSE loss.
        """
        losses = {}
        for rank in pred:
            p = pred[rank]
            r = ref[str(rank)]

            ratio = self.loss_hparams["ratio"][rank]
            losses[f"train/loss_rank-{rank}"] = ratio * self.loss_func(p, r)

        losses["train/loss_total"] = sum(losses.values())

        return losses

    def compute_loss_cart(self, pred: Tensor, ref: Tensor):
        return {"train/loss_total": self.loss_func(pred, ref)}

    def compute_metrics_nat(self, pred: dict, ref: dict, mode: str):
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
            metrics[f"{mode}/{self.metrics_type}_rank-{rank}"] = self.metrics[name]

        return metrics

    def compute_metrics_cart(self, pred: Tensor, ref: Tensor, mode: str):
        name = f"metrics_{mode}_cartesian"
        self.metrics_cartesian[name](pred, ref)

        return {f"{mode}/{self.metrics_type}_cartesian": self.metrics_cartesian[name]}


AtomicTensorLitModule = StructureTensorLitModule
