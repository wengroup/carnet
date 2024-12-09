"""PyTorch Lightning Trainer."""
from typing import Any

import torch
from ema_pytorch import EMA
from lightning import LightningModule
from lightning.pytorch.cli import instantiate_class
from lightning.pytorch.utilities import grad_norm
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torch import nn
from torchmetrics import MeanAbsoluteError, MeanSquaredError

from carten.data.utils import get_edge_vec
from carten.model.force_stress import compute_forces


class LitModel(LightningModule):
    def __init__(
        self,
        model: nn.Module,
        loss_hparams: dict[str, float] = None,
        metrics_hparams: dict[str, float] = None,
        optimizer_hparams: dict[str, Any] = None,
        lr_scheduler_hparams: dict[str, Any] = None,
        ema_hparams: dict[str, Any] = None,
        other_hparams: dict[str, Any] = None,
    ):
        """
        Args:
            model: PyTorch model.
            other_hparams: other hyperparameters, which are not used anywhere within
                but are passed in such that they can be logged automatically.

        """
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

        self.validation_start_epoch = self.metrics_hparams.get(
            "validation_start_epoch", 0
        )

        self.energy_metric = nn.ModuleDict(
            {
                f"metrics_{mode}": MetricClass()
                for mode in ["train", "val", "test", "val_ema", "test_ema"]
            }
        )

        self.forces_metric = nn.ModuleDict(
            {
                f"metrics_{mode}": MetricClass()
                for mode in ["train", "val", "test", "val_ema", "test_ema"]
            }
        )

    def forward(self, batch):
        """Compute energy and forces."""
        # Note, it is tempting to compute the edge_vector in the collate_fn of the
        # dataloader. However, this will not work will pytorch lightning, internally
        # it does something to the batch, modifying both pos and edge_index. As a
        # result, the edge_vector is not directly derived from pos, and thus we won't
        # be able to compute the forces.
        # This is why we compute the edge_vector here.

        e_pred = self.model(
            edge_vector=self._get_edge_vector(batch),
            edge_idx=batch.edge_index,
            atom_type=batch.atom_type,
            num_atoms=batch.num_atoms,
        )
        f_pred = compute_forces(e_pred, batch.pos, self.training)

        return e_pred, f_pred

    def forward_ema(self, batch):
        """Same as `forward, but use the EMA model instead of the original model."""

        e_pred = self.ema(
            edge_vector=self._get_edge_vector(batch),
            edge_idx=batch.edge_index,
            atom_type=batch.atom_type,
            num_atoms=batch.num_atoms,
        )
        f_pred = compute_forces(e_pred, batch.pos, self.training)

        return e_pred, f_pred

    def training_step(self, batch, batch_idx):
        # requires_grad to enable force computation
        batch.pos.requires_grad_(True)

        e_ref = batch.y["energy"]
        f_ref = batch.y["forces"]
        num_atoms = batch.num_atoms
        batch_size = batch.num_graphs

        e_pred, f_pred = self(batch)

        if self.loss_hparams["normalize"]:
            # losses = self.compute_loss_1(e_pred, f_pred, e_ref, f_ref, num_atoms)
            losses = self.compute_loss_1_2(e_pred, f_pred, e_ref, f_ref, num_atoms)
        else:
            # losses = self.compute_loss_2(e_pred, f_pred, e_ref, f_ref, num_atoms)
            losses = self.compute_loss_2_2(e_pred, f_pred, e_ref, f_ref, num_atoms)
        metrics = self.compute_metrics(e_pred, f_pred, e_ref, f_ref, num_atoms, "train")
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
        with torch.enable_grad():
            # requires_grad to enable force computation
            batch.pos.requires_grad_(True)

            e_ref = batch.y["energy"]
            f_ref = batch.y["forces"]
            num_atoms = batch.num_atoms
            batch_size = batch.num_graphs

            # use current model
            if self.current_epoch >= start_epoch:
                e_pred, f_pred = self(batch)
            else:
                # dummy values to skip validation for the first few epochs
                e_pred = f_pred = torch.zeros(1, dtype=e_ref.dtype, device=e_ref.device)
                e_ref = f_ref = 1e10 * torch.ones(
                    1, dtype=e_ref.dtype, device=e_ref.device
                )

            metrics = self.compute_metrics(
                e_pred, f_pred, e_ref, f_ref, num_atoms, mode
            )
            self.log_dict(
                metrics,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                batch_size=batch_size,
            )

            # use EMA model
            if self.current_epoch >= start_epoch:
                e_pred, f_pred = self.forward_ema(batch)
            else:
                # dummy values to skip validation for the first few epochs
                e_pred = f_pred = torch.zeros(1, dtype=e_ref.dtype, device=e_ref.device)
                e_ref = f_ref = 1e10 * torch.ones(
                    1, dtype=e_ref.dtype, device=e_ref.device
                )

            metrics = self.compute_metrics(
                e_pred, f_pred, e_ref, f_ref, num_atoms, mode + "_ema"
            )
            self.log_dict(
                metrics,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                batch_size=batch_size,
            )

    def compute_loss_1(self, e_pred, f_pred, e_ref, f_ref, num_atoms):
        """
        This loss
            - normalize energy by N_K^2
            - normalize forces by N_K

        It treats each structure equally.

        This is used in:
            - `structure` model of MTP
            - Allegro (eq 30)
            - MACE foundation model, eq 1 of https://doi.org/10.48550/arXiv.2401.00096

            for Allegro and MACE, it is not quite that. They only divide the total
            forces by the total number of atoms. Not considering individual structures.

        L = 1/K \sum_k^K [
                1/N_k^2 (E_pred - E_ref)^2
              + 1/(3N_k) \sum_i^{N_k} (F_pred - F_ref)^2
              ]

        where N_k is the number of atoms in the k-th structure
        """
        e_loss = self.loss_hparams["energy_ratio"] * nn.functional.mse_loss(
            e_pred / num_atoms, e_ref / num_atoms, reduction="mean"
        )

        f_norm = torch.repeat_interleave(num_atoms, num_atoms).sqrt().reshape(-1, 1)
        K = len(num_atoms)  # batch size
        f_loss = (
            self.loss_hparams["forces_ratio"]
            * nn.functional.mse_loss(f_pred / f_norm, f_ref / f_norm, reduction="sum")
            / (3 * K)
        )

        loss = e_loss + f_loss

        losses = {
            "train/loss_total": loss,
            "train/loss_energy": e_loss,
            "train/loss_forces": f_loss,
        }

        return losses

    def compute_loss_1_2(self, e_pred, f_pred, e_ref, f_ref, num_atoms):
        """
        A simplified version of `compute_loss_1` that only works for dataset with the
        same number of atoms in each config, i.e. N_k = N for all k.

        In this case, the force term is simply:
            L_f = 1/(3KN) \sum_k \sum_i (F_pred - F_ref)^2
        """

        e_loss = self.loss_hparams["energy_ratio"] * nn.functional.mse_loss(
            e_pred / num_atoms, e_ref / num_atoms, reduction="mean"
        )
        f_loss = self.loss_hparams["forces_ratio"] * nn.functional.mse_loss(
            f_pred, f_ref, reduction="mean"
        )
        loss = e_loss + f_loss

        losses = {
            "train/loss_total": loss,
            "train/loss_energy": e_loss,
            "train/loss_forces": f_loss,
        }

        return losses

    def compute_loss_2(self, e_pred, f_pred, e_ref, f_ref, num_atoms):
        """
        This loss
            - not normalize energy
            - normalize forces by N_k

            L = 1/K \sum_k^K [
                (E_pred - E_ref)^2
              + 1/(3N_k) \sum_i^{N_k} (F_pred - F_ref)^2
              ]

        This is the used in
            - NequIP (eq 9)
            - Allegro (eq 29)
            - MACE (eq 15) https://doi.org/10.48550/arXiv.2206.07697
        """

        e_loss = self.loss_hparams["energy_ratio"] * nn.functional.mse_loss(
            e_pred, e_ref, reduction="mean"
        )

        K = len(num_atoms)  # batch size
        f_norm = torch.repeat_interleave(num_atoms, num_atoms).sqrt().reshape(-1, 1)
        f_loss = (
            self.loss_hparams["forces_ratio"]
            * nn.functional.mse_loss(f_pred / f_norm, f_ref / f_norm, reduction="sum")
            / (3 * K)
        )

        loss = e_loss + f_loss

        losses = {
            "train/loss_total": loss,
            "train/loss_energy": e_loss,
            "train/loss_forces": f_loss,
        }

        return losses

    def compute_loss_2_2(self, e_pred, f_pred, e_ref, f_ref, num_atoms):
        """
        A simplified version of `compute_loss_2` that only works for dataset with the
        same number of atoms in each config, i.e. N_k = N for all k.

        Note, division by K for energy (3KN for forces) is considered with  `mse_loss`,
        by using `reduction=mean`.
        """

        e_loss = self.loss_hparams["energy_ratio"] * nn.functional.mse_loss(
            e_pred, e_ref, reduction="mean"
        )
        f_loss = self.loss_hparams["forces_ratio"] * nn.functional.mse_loss(
            f_pred, f_ref, reduction="mean"
        )
        loss = e_loss + f_loss

        losses = {
            "train/loss_total": loss,
            "train/loss_energy": e_loss,
            "train/loss_forces": f_loss,
        }

        return losses

    def compute_loss_3(self, e_pred, f_pred, e_ref, f_ref, num_atoms):
        """
        This loss
            - normalize energy by N
            - not normalize forces by N

        L = 1/K \sum_k^K [ 1/N_k^2 (E_pred - E_ref)^2 }
                         + \sum_i^{N_k} (F_pred - F_ref)^2 ]


        This is used in
            - the `vibrations` model of MTP
            - a MACE model Eq (14) of https://doi.org/10.1063/5.0155322
        """

    def compute_metrics(self, e_pred, f_pred, e_ref, f_ref, num_atoms, mode: str):
        """
        Energy MAE:
            MAE = \mean_k |E_pred/N_k - E_ref/N_k|

            or

            MAE = \mean_k |E_pred - E_ref|

        Forces MAE:
            MAE = \mean_k 1/(3N_k) \sum_i |F_pred - F_ref|


        """
        if self.metrics_hparams["normalize"]:
            self.energy_metric[f"metrics_{mode}"](e_pred / num_atoms, e_ref / num_atoms)
        else:
            self.energy_metric[f"metrics_{mode}"](e_pred, e_ref)

        # NOTE, the below will bias the MAE towards larger structures, because each
        # atom contributes to the MAE.
        # If we want to have a per-structure MAE, we should compute the MAE for each
        # structure separately, and then average over the structures.
        self.forces_metric[f"metrics_{mode}"](f_pred, f_ref)

        metrics = {
            f"{mode}/{self.metrics_type}_e": self.energy_metric[f"metrics_{mode}"],
            f"{mode}/{self.metrics_type}_f": self.forces_metric[f"metrics_{mode}"],
        }

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
