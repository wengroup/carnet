"""Lightning model for predicting tensorial properties of materials and molecules.

This is supposed to be used with the `carten.model.tensor.StructureTensorModel`
and `carten.model.tensor.AtomicTensorModel` models.
"""

from typing import Any

import torch
from ema_pytorch import EMA
from lightning import LightningModule
from lightning.pytorch.cli import instantiate_class
from torch import Tensor, nn
from torchmetrics import MeanAbsoluteError, MeanSquaredError

from carten.core.convert import Converter
from carten.data.utils import get_edge_vec

from ..force_stress import compute_forces


class MultiTaskLitModule(LightningModule):
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

        self.metrics = nn.ModuleDict(
            {
                f"metrics_{mode}_{target}": MetricClass()
                for mode in ["train", "val", "test", "val_ema", "test_ema"]
                for target in self.loss_hparams["target_name"]
            }
        )

        # values needed to convert natural tensors to cartesian tensors
        self.converter = nn.ModuleDict(
            {
                target: Converter(sym)
                for target, sym in self.loss_hparams["target_symmetry"].items()
            }
        )

    def forward(self, batch):
        """Compute model output."""

        # requires_grad to enable force computation
        batch.pos.requires_grad_(True)

        # Atomic selector only needed for atomic tensor model, not for structure tensor
        # model.
        atomic_selector = (
            None if "atomic_selector" not in batch.y else batch.y["atomic_selector"]
        )

        output = self.model(
            edge_vector=self._get_edge_vector(batch),
            edge_idx=batch.edge_index,
            atom_type=batch.atom_type,
            num_atoms=batch.num_atoms,
            atomic_selector=atomic_selector,
        )
        output["forces"] = compute_forces(output["energy"], batch.pos, self.training)

        return output

    def forward_ema(self, batch):
        """Same as `forward, but use the EMA model instead of the original model."""

        # requires_grad to enable force computation
        batch.pos.requires_grad_(True)

        atomic_selector = (
            None if "atomic_selector" not in batch.y else batch.y["atomic_selector"]
        )
        output = self.ema(
            edge_vector=self._get_edge_vector(batch),
            edge_idx=batch.edge_index,
            atom_type=batch.atom_type,
            num_atoms=batch.num_atoms,
            atomic_selector=atomic_selector,
        )

        output["forces"] = compute_forces(output["energy"], batch.pos, self.training)

        return output

    def training_step(self, batch, batch_idx):

        batch_size = batch.num_graphs

        # References
        ref = batch.y

        # deal with shield tensor shape (we have multiple atoms predicted together)
        name = "shielding_tensor_full"
        if name in ref:
            ref[name] = ref[name].view(-1, 3, 3)

        pred_nat = self(batch)
        pred = self.to_cartesian(pred_nat)

        losses = self.compute_losses(pred, ref)
        metrics = self.compute_metrics(pred, ref, mode="train")

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
        with torch.enable_grad():

            batch_size = batch.num_graphs

            # References
            ref = batch.y
            # deal with shield tensor shape (we have multiple atoms predicted together)
            name = "shielding_tensor_full"
            if name in ref:
                ref[name] = ref[name].view(-1, 3, 3)

            pred_nat = self(batch)
            pred = self.to_cartesian(pred_nat)

            metrics = self.compute_metrics(pred, ref, mode)

            # use EMA model
            pred_nat = self.forward_ema(batch)
            pred = self.to_cartesian(pred_nat)

            metrics_eam = self.compute_metrics(pred, ref, mode + "_ema")
            metrics.update(metrics_eam)

            self.log_dict(
                metrics,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                batch_size=batch_size,
            )

    def compute_losses(self, pred: dict, ref: dict):
        losses = {}
        total = 0
        for target in self.loss_hparams["target_name"]:

            if "tensor" in target:
                l = nn.functional.mse_loss(
                    pred[target + "_full"], ref[target + "_full"], reduction="mean"
                )
            else:
                l = nn.functional.mse_loss(pred[target], ref[target], reduction="mean")

            weight = self.loss_hparams["ratio"][target]
            l *= weight

            losses[f"train/loss_{target}"] = l

            total += l

        losses["train/loss_total"] = total

        return losses

    def compute_metrics(self, pred: dict, ref: dict, mode: str = "train"):

        metrics = {}
        for target in self.loss_hparams["target_name"]:
            name = f"metrics_{mode}_{target}"

            # Change the name for tensors, not `energy` or `forces`
            if "tensor" in target:
                target = target + "_full"

            self.metrics[name](pred[target], ref[target])
            metrics[f"{mode}/mae_{target}"] = self.metrics[name]

        return metrics

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
            shift_vec = batch.shift_vector
        except AttributeError:
            # this happens for molecules, no pbc needed, and no cell
            cell = None
            shift_vec = None

        edge_vector = get_edge_vec(
            batch.pos, shift_vec, cell, batch.edge_index, batch.batch
        )

        return edge_vector

    def to_cartesian(self, pred_nat: dict[str, Tensor]) -> dict[str, Tensor]:
        """Convert natural tensors to cartesian tensors.

        Args:
            pred_nat (dict[str, Tensor]): Predictions in natural tensor format.

        Returns:
            dict[str, Tensor]: Predictions in cartesian tensor format.
        """
        pred = {"energy": pred_nat["energy"], "forces": pred_nat["forces"]}

        for target, _ in self.loss_hparams["target_signature"].items():
            x = pred_nat[target]

            # Need conversion
            if target in self.loss_hparams["target_symmetry"]:
                pred[target + "_full"] = self.converter[target].to_ordinary_tensor(
                    x, flatten_tensor_dim=False
                )
            # Does not need conversion
            else:
                # TODO, hardcode it for dipole moment tensor, which needs no converter
                if target == "dipole_moment_tensor":
                    x = x[1].reshape(
                        -1, 3
                    )  # select rank-1 tensor and make it the correct shape
                    pred[target + "_full"] = x
                else:
                    raise NotImplementedError(
                        f"Conversion for target {target} not implemented."
                    )

        return pred
