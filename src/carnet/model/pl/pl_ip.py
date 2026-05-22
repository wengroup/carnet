"""Lightning model for interatomic potential.

This is supposed to be used with the `carnet.model.ip.InteratomicPotential` model.
"""

from typing import Any

import torch
from ema_pytorch import EMA
from lightning import LightningModule
from lightning.pytorch.cli import instantiate_class
from torch import nn
from torchmetrics import MeanAbsoluteError, MeanSquaredError

from carnet.data.data import get_edge_vec_batch
from carnet.model.force_stress import apply_strain

from ..force_stress import compute_forces, compute_forces_stress


class InteratomicPotentialLitModule(LightningModule):
    def __init__(
        self,
        model: nn.Module,
        loss_hparams: dict[str, float] = None,
        metrics_hparams: dict[str, float] = None,
        ema_hparams: dict[str, Any] = None,
        optimizer_hparams: dict[str, Any] = None,
        lr_scheduler_hparams: dict[str, Any] = None,
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
        self.ema_hparams = ema_hparams
        self.optimizer_hparams = optimizer_hparams
        self.lr_scheduler_hparams = lr_scheduler_hparams

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

        self.need_stress = self.loss_hparams.get("stress_ratio", 0.0) > 1e-6

        if self.need_stress:
            self.stress_metric = nn.ModuleDict(
                {
                    f"metrics_{mode}": MetricClass()
                    for mode in ["train", "val", "test", "val_ema", "test_ema"]
                }
            )
        else:
            self.stress_metric = None

    def forward(self, batch):
        return self._forward(batch, self.model)

    def forward_ema(self, batch):
        """Same as `forward, but use the EMA model instead of the original model."""
        return self._forward(batch, self.ema)

    def _forward(self, batch, func):
        """Compute energy, forces, and stress."""

        # Note: edge_vector must be computed here rather than in the DataLoader.
        # Computing it earlier would break the autograd graph during multiprocessing
        # serialization and data transfer. As a result, edge_vector becomes leaf nodes
        # in the autograd graph (instead of being connected to `pos`), preventing the
        # computation of forces via gradients of energy w.r.t. positions.
        # We compute it here after setting `requires_grad` to ensure forces (gradients)
        # can be correctly derived.

        # requires_grad to enable force computation
        pos = batch.pos
        pos.requires_grad_(True)

        cell = batch.cell if hasattr(batch, "cell") else None
        if cell is not None:
            cell = cell.reshape(-1, 3, 3)  # (B*3, 3) -> (B, 3, 3)

        # Need stress calculation
        if self.need_stress:
            if cell is not None:
                # apply strain and get strained positions and cell
                strain, strained_pos, strained_cell = apply_strain(
                    pos, cell, batch.batch
                )
                pos = strained_pos
                cell = strained_cell
            else:
                raise RuntimeError("Need stress but cell is not provided in the batch.")
        else:
            strain = None

        edge_vector = get_edge_vec_batch(
            pos, batch.shift_vector, cell, batch.edge_index, batch.batch
        )

        e_pred, _ = func(
            edge_vector=edge_vector,
            edge_idx=batch.edge_index,
            atom_type=batch.atom_type,
            num_atoms=batch.num_atoms,
            atomic_number=batch.atomic_number,
            batch=batch.batch,
        )

        if self.need_stress:
            f_pred, s_pred = compute_forces_stress(
                e_pred, pos, cell, strain, self.training
            )

        else:
            f_pred = compute_forces(e_pred, pos, self.training)
            s_pred = None

        return e_pred, f_pred, s_pred

    def training_step(self, batch, batch_idx):

        e_ref = batch.y["energy"]
        f_ref = batch.y["forces"]
        s_ref = batch.y.get("stress", None)

        num_atoms = batch.num_atoms
        batch_size = batch.num_graphs

        e_pred, f_pred, s_pred = self(batch)

        # select only the samples that have stress labels
        if self.need_stress:
            s_ref = s_ref.reshape(-1, 3, 3)  # (B*3, 3) -> (B, 3, 3)

            # bool to indicate whether a config has stress, shape (B,)
            has_stress = batch.y.get("has_stress", None)
            if has_stress is not None:
                s_pred = s_pred[has_stress]
                s_ref = s_ref[has_stress]

        losses = self.compute_loss(
            e_pred, f_pred, s_pred, e_ref, f_ref, s_ref, num_atoms
        )
        metrics = self.compute_metrics(
            e_pred, f_pred, s_pred, e_ref, f_ref, s_ref, num_atoms, "train"
        )
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
            e_ref = batch.y["energy"]
            f_ref = batch.y["forces"]
            s_ref = batch.y.get("stress", None)

            num_atoms = batch.num_atoms
            batch_size = batch.num_graphs

            # use current model
            e_pred, f_pred, s_pred = self(batch)

            # select only the samples that have stress labels
            if self.need_stress:
                s_ref = s_ref.reshape(-1, 3, 3)  # (B*3, 3) -> (B, 3, 3)

                # bool to indicate whether a config has stress, shape (B,)
                has_stress = batch.y.get("has_stress", None)
                if has_stress is not None:
                    s_pred = s_pred[has_stress]
                    s_ref = s_ref[has_stress]

            metrics = self.compute_metrics(
                e_pred, f_pred, s_pred, e_ref, f_ref, s_ref, num_atoms, mode
            )

            # use EMA model
            e_pred_ema, f_pred_ema, s_pred_ema = self.forward_ema(batch)

            # select only the samples that have stress labels
            if self.need_stress:
                # No need to deal with s_ref, already processed above
                # s_ref = s_ref.reshape(-1, 3, 3)  # (B*3, 3) -> (B, 3, 3)

                # bool to indicate whether a config has stress, shape (B,)
                has_stress = batch.y.get("has_stress", None)
                if has_stress is not None:
                    s_pred = s_pred[has_stress]
                    # s_ref = s_ref[has_stress]

            metrics_ema = self.compute_metrics(
                e_pred_ema,
                f_pred_ema,
                s_pred_ema,
                e_ref,
                f_ref,
                s_ref,
                num_atoms,
                mode + "_ema",
            )
            metrics.update(metrics_ema)

            self.log_dict(
                metrics,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                batch_size=batch_size,
            )

    def compute_loss(self, e_pred, f_pred, s_pred, e_ref, f_ref, s_ref, num_atoms):
        """

        Energy:

        If not normalize:
            L_e = (1/Nc) * L(E_pred, E_ref)

        If normalize:
            L_e = (1/Nc) * L(E_pred/Nc, E_ref/Nc)

        Forces:
            L_f = (1/Nf) * L(F_pred, F_ref)

        Stress: (if needed)
            L_s = (1/(9Nc)) * L(S_pred, S_ref)

        where Nc is the number of configurations in the batch, Nf is the total number
        of forces components in the batch (3 * total number of atoms in the batch).

        Note, if the batch consists of configurations of different sizes, normalizing
        the energy by the number of atoms ensures that the energy is treated on a
        per-atom basis. In this case, it can avoid the situation where larger structures
        dominate the energy loss.
        For forces and stress, they system size normalization is already taken into
        account.

        If we want to treat each structure equally, use `compute_loss_1` instead.
        """
        if self.loss_hparams["normalize"]:
            # e_pred and e_ref are (B,), num_atoms is (B,)
            e_pred = e_pred / num_atoms
            e_ref = e_ref / num_atoms

        e_loss = self.loss_hparams["energy_ratio"] * self.loss_func(e_pred, e_ref)
        f_loss = self.loss_hparams["forces_ratio"] * self.loss_func(f_pred, f_ref)

        loss = e_loss + f_loss
        losses = {"train/loss_energy": e_loss, "train/loss_forces": f_loss}

        if self.need_stress:
            s_loss = self.loss_hparams["stress_ratio"] * self.loss_func(s_pred, s_ref)

            loss += s_loss
            losses["train/loss_stress"] = s_loss

        losses["train/loss_total"] = loss

        return losses

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

        L = 1/K sum_k^K [
                1/N_k^2 (E_pred - E_ref)^2
              + 1/(3N_k) sum_i^{N_k} (F_pred - F_ref)^2
              ]

        where N_k is the number of atoms in the k-th structure
        """
        # e_pred and e_ref are (B,), num_atoms is (B,)
        e_pred = e_pred / num_atoms
        e_ref = e_ref / num_atoms
        e_loss = self.loss_hparams["energy_ratio"] * self.loss_func(
            e_pred, e_ref, reduction="mean"
        )

        f_norm = torch.repeat_interleave(num_atoms, num_atoms).sqrt().reshape(-1, 1)
        K = len(num_atoms)  # batch size
        f_loss = (
            self.loss_hparams["forces_ratio"]
            * self.loss_func(f_pred / f_norm, f_ref / f_norm, reduction="sum")
            / (3 * K)
        )

        loss = e_loss + f_loss

        losses = {
            "train/loss_total": loss,
            "train/loss_energy": e_loss,
            "train/loss_forces": f_loss,
        }

        return losses

    def compute_metrics(
        self, e_pred, f_pred, s_pred, e_ref, f_ref, s_ref, num_atoms, mode: str
    ):
        """
        Energy MAE:
            MAE = mean_k |E_pred/N_k - E_ref/N_k|

            or

            MAE = mean_k |E_pred - E_ref|

        Forces MAE:
            MAE = mean_k 1/(3N_k) sum_i |F_pred - F_ref|


        """
        if self.metrics_hparams["normalize"]:
            # e_pred and e_ref are (B,), num_atoms is (B,)
            e_pred = e_pred / num_atoms
            e_ref = e_ref / num_atoms

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

        if self.need_stress:
            self.stress_metric[f"metrics_{mode}"](s_pred, s_ref)
            metrics[f"{mode}/{self.metrics_type}_s"] = self.stress_metric[
                f"metrics_{mode}"
            ]

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
                        "ReduceLROnPlateau."
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
