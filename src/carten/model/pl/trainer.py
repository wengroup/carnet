import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
from ema_pytorch import EMA
from line_profiler import profile
from torch.utils.data import DataLoader
from torchmetrics import MeanAbsoluteError, MeanSquaredError
from tqdm import tqdm

from carten.core.convert import Converter
from carten.model.elastic import get_voigt_projection_tensor


class BasePyTorchTrainer:
    """
    Base PyTorch trainer for predicting tensorial properties of materials and molecules.

    This should not be used directly, but should be subclassed by the specific models.
    """

    def __init__(
        self,
        model: nn.Module,
        loss_hparams: Dict[str, Any] = None,
        metrics_hparams: Dict[str, float] = None,
        optimizer_hparams: Dict[str, Any] = None,
        lr_scheduler_hparams: Dict[str, Any] = None,
        ema_hparams: Dict[str, Any] = None,
        other_hparams: Dict[str, Any] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.device = torch.device(device)
        self.model = model.to(self.device)

        self.loss_hparams = loss_hparams or {}
        self.metrics_hparams = metrics_hparams or {}
        self.optimizer_hparams = optimizer_hparams or {}
        self.lr_scheduler_hparams = lr_scheduler_hparams or {}
        self.ema_hparams = ema_hparams or {}
        self.other_hparams = other_hparams or {}

        # Initialize EMA
        self.ema = EMA(self.model, **self.ema_hparams)

        # Setup metrics
        self.metrics_type = self.metrics_hparams.get("type", "mae")
        if self.metrics_type == "mae":
            MetricClass = MeanAbsoluteError
        elif self.metrics_type == "mse":
            MetricClass = MeanSquaredError
        else:
            raise ValueError(f"Unknown metrics type: {self.metrics_type}")

        # Validation start epoch
        self.validation_start_epoch = self.metrics_hparams.get(
            "validation_start_epoch", 0
        )

        # Initialize metrics for natural tensors
        self.metrics = nn.ModuleDict(
            {
                f"metrics_{mode}_{rank}": MetricClass().to(self.device)
                for mode in ["train", "val", "test", "val_ema", "test_ema"]
                for rank in self.loss_hparams["target_signature"]
            }
        )

        # Setup target mode and cartesian tensor handling
        self.target_mode = self.loss_hparams.get("target_mode", "natural")

        if self.target_mode == "natural":
            self.metrics_cartesian = None
            self.converter = None
        elif self.target_mode in ["full", "voigt"]:
            self.metrics_cartesian = nn.ModuleDict(
                {
                    f"metrics_{mode}_cartesian": MetricClass().to(self.device)
                    for mode in ["train", "val", "test", "val_ema", "test_ema"]
                }
            )

            # Setup converter
            symmetry = self.loss_hparams.get("target_symmetry")
            if symmetry is None:
                raise ValueError('"target_symmetry" must be provided in loss_hparams.')
            self.converter = Converter(symmetry).to(self.device)
        else:
            raise ValueError(f"Unknown target mode: {self.target_mode}")

        # Setup Voigt projection tensor if needed
        if self.target_mode == "voigt":
            M = get_voigt_projection_tensor()
            M = M.reshape(-1, 6 * 6)
            self.M = M.to(self.device)
        else:
            self.M = None

        # Setup optimizer and scheduler
        self.optimizer = self._setup_optimizer()
        self.lr_scheduler = self._setup_lr_scheduler()

        # Training state
        self.current_epoch = 0
        self.global_step = 0

        # Setup logging
        self.logger = logging.getLogger(__name__)

    def _setup_optimizer(self):
        """Setup optimizer from hyperparameters."""

        # Parse class path and init args
        class_path = self.optimizer_hparams.get("class_path")
        init_args = self.optimizer_hparams.get("init_args", {})

        # Get optimizer class
        module_path, class_name = class_path.rsplit(".", 1)
        module = __import__(module_path, fromlist=[class_name])
        optimizer_class = getattr(module, class_name)

        return optimizer_class(self.model.parameters(), **init_args)

    def _setup_lr_scheduler(self):
        """Setup learning rate scheduler from hyperparameters."""
        class_path = self.lr_scheduler_hparams.get("class_path")
        if class_path is None or class_path == "none":
            return None

        init_args = self.lr_scheduler_hparams.get("init_args", {})

        # Get scheduler class
        module_path, class_name = class_path.rsplit(".", 1)
        module = __import__(module_path, fromlist=[class_name])
        scheduler_class = getattr(module, class_name)

        return scheduler_class(self.optimizer, **init_args)

    @profile
    def forward(self, batch):
        """Compute model output."""
        atomic_selector = (
            None if "atomic_selector" not in batch.y else batch.y["atomic_selector"]
        )

        return self.model(
            edge_vector=batch.edge_vector,
            edge_idx=batch.edge_index,
            atom_type=batch.atom_type,
            num_atoms=batch.num_atoms,
            atomic_selector=atomic_selector,
        )

    @profile
    def forward_ema(self, batch):
        """Same as forward, but use the EMA model instead of the original model."""
        atomic_selector = (
            None if "atomic_selector" not in batch.y else batch.y["atomic_selector"]
        )

        return self.ema(
            edge_vector=batch.edge_vector,
            edge_idx=batch.edge_index,
            atom_type=batch.atom_type,
            num_atoms=batch.num_atoms,
            atomic_selector=atomic_selector,
        )

    @profile
    def training_step(self, batch):
        """Single training step."""
        self.model.train()
        batch_size = batch.num_graphs

        # Move batch to device
        batch = batch.to(self.device)

        ref_nat = batch.y[self.loss_hparams["target_name"] + "_natural"]
        pred_nat = self.forward(batch)

        # Compute loss and metrics based on target mode
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

        # Backward pass
        self.optimizer.zero_grad()
        losses["train/loss_total"].backward()
        self.optimizer.step()

        # Update EMA
        self.ema.update()

        # Update global step
        self.global_step += 1

        return {**losses, **metrics, "batch_size": batch_size}

    @profile
    def validation_step(self, batch):
        """Single validation step."""
        return self._val_test_step(
            batch, mode="val", start_epoch=self.validation_start_epoch
        )

    @profile
    def test_step(self, batch):
        """Single test step."""
        return self._val_test_step(batch, mode="test")

    def _val_test_step(self, batch, mode: str, start_epoch: int = 0):
        """Common validation/test step logic."""
        self.model.eval()
        batch_size = batch.num_graphs

        # Move batch to device
        batch = batch.to(self.device)

        ref_nat = batch.y[self.loss_hparams["target_name"] + "_natural"]
        ref = batch.y[
            self.loss_hparams["target_name"] + "_" + self.loss_hparams["target_mode"]
        ]

        with torch.no_grad():
            # Use current model
            if self.current_epoch >= start_epoch:
                pred_nat = self.forward(batch)
            else:
                # Dummy values to skip validation for the first few epochs
                pred_nat = {int(k): v + 1e10 for k, v in ref_nat.items()}

            if self.target_mode in ["full", "voigt"]:
                pred = self.to_cartesian(pred_nat)
            else:
                pred = None

            metrics = self.compute_metrics(pred_nat, ref_nat, pred, ref, mode)

            # Use EMA model
            if self.current_epoch >= start_epoch:
                pred_nat_ema = self.forward_ema(batch)
            else:
                # Dummy values to skip validation for the first few epochs
                pred_nat_ema = {int(k): v + 1e5 for k, v in ref_nat.items()}

            if self.target_mode in ["full", "voigt"]:
                pred_ema = self.to_cartesian(pred_nat_ema)
            else:
                pred_ema = None

            metrics_ema = self.compute_metrics(
                pred_nat_ema, ref_nat, pred_ema, ref, mode + "_ema"
            )
            metrics.update(metrics_ema)

        return {**metrics, "batch_size": batch_size}

    @profile
    def train_epoch(self, train_loader: DataLoader):
        """Train for one epoch."""
        self.model.train()
        epoch_metrics = []

        pbar = tqdm(train_loader, desc=f"Epoch {self.current_epoch} [Train]")
        for batch in pbar:
            step_metrics = self.training_step(batch)
            epoch_metrics.append(step_metrics)

            # Update progress bar
            loss_val = step_metrics["train/loss_total"].item()
            pbar.set_postfix({"loss": f"{loss_val:.4f}"})

        return self._aggregate_metrics(epoch_metrics)

    @profile
    def validate_epoch(self, val_loader: DataLoader):
        """Validate for one epoch."""
        self.model.eval()
        epoch_metrics = []

        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f"Epoch {self.current_epoch} [Val]")
            for batch in pbar:
                step_metrics = self.validation_step(batch)
                epoch_metrics.append(step_metrics)

        return self._aggregate_metrics(epoch_metrics)

    @profile
    def test_epoch(self, test_loader: DataLoader):
        """Test for one epoch."""
        self.model.eval()
        epoch_metrics = []

        with torch.no_grad():
            pbar = tqdm(test_loader, desc="Testing")
            for batch in pbar:
                step_metrics = self.test_step(batch)
                epoch_metrics.append(step_metrics)

        return self._aggregate_metrics(epoch_metrics)

    def _aggregate_metrics(self, epoch_metrics):
        """Aggregate metrics across batches."""
        aggregated = {}
        total_batch_size = sum(m["batch_size"] for m in epoch_metrics)

        for key in epoch_metrics[0]:
            if key == "batch_size":
                continue

            if isinstance(epoch_metrics[0][key], torch.Tensor):
                # For scalar metrics, compute weighted average
                if epoch_metrics[0][key].numel() == 1:
                    weighted_sum = sum(m[key] * m["batch_size"] for m in epoch_metrics)
                    aggregated[key] = (weighted_sum / total_batch_size).item()
                else:
                    # For tensor metrics, let the metric object handle aggregation
                    aggregated[key] = epoch_metrics[-1][key]
            else:
                # For metric objects, compute the final value
                aggregated[key] = epoch_metrics[-1][key].compute().item()

        return aggregated

    @profile
    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        max_epochs: int = 100,
        checkpoint_dir: Optional[str] = None,
    ):
        """Main training loop."""
        if checkpoint_dir:
            checkpoint_dir = Path(checkpoint_dir)
            checkpoint_dir.mkdir(parents=True, exist_ok=True)

        best_val_loss = float("inf")

        for epoch in range(max_epochs):
            self.current_epoch = epoch

            # Training
            train_metrics = self.train_epoch(train_loader)
            self.logger.info(f"Epoch {epoch} - Train metrics: {train_metrics}")

            # Validation
            if val_loader is not None:
                val_metrics = self.validate_epoch(val_loader)
                self.logger.info(f"Epoch {epoch} - Val metrics: {val_metrics}")

                # Learning rate scheduling
                if self.lr_scheduler is not None:
                    if isinstance(
                        self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
                    ):
                        monitor_key = self.lr_scheduler_hparams.get(
                            "monitor", "val/loss_total"
                        )
                        if monitor_key in val_metrics:
                            self.lr_scheduler.step(val_metrics[monitor_key])
                    else:
                        self.lr_scheduler.step()

                # Checkpointing
                if checkpoint_dir:
                    val_loss = val_metrics.get("val_ema/mae_cartesian", float("inf"))
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        self.save_checkpoint(checkpoint_dir / "best_model.pth")

                    # TODO,
                    # # Save regular checkpoint
                    # if epoch % 10 == 0:
                    #     self.save_checkpoint(checkpoint_dir / f"epoch_{epoch}.pth")

            # Reset metrics for next epoch
            self._reset_metrics()

    @profile
    def test(self, test_loader: DataLoader, checkpoint_path: Optional[str] = None):
        """Test the model."""
        if checkpoint_path:
            self.load_checkpoint(checkpoint_path)

        test_metrics = self.test_epoch(test_loader)

        self.logger.info(f"Test metrics: {test_metrics}")

        return test_metrics

    def _reset_metrics(self):
        """Reset all metrics for the next epoch."""
        for metric in self.metrics.values():
            metric.reset()

        if self.metrics_cartesian is not None:
            for metric in self.metrics_cartesian.values():
                metric.reset()

    def save_checkpoint(self, path: str):
        """Save model checkpoint."""
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "ema_state_dict": self.ema.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "current_epoch": self.current_epoch,
            "global_step": self.global_step,
        }

        if self.lr_scheduler is not None:
            checkpoint["lr_scheduler_state_dict"] = self.lr_scheduler.state_dict()

        torch.save(checkpoint, path)
        self.logger.info(f"Checkpoint saved to {path}")

    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.ema.load_state_dict(checkpoint["ema_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.current_epoch = checkpoint["current_epoch"]
        self.global_step = checkpoint["global_step"]

        if self.lr_scheduler is not None and "lr_scheduler_state_dict" in checkpoint:
            self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])

        self.logger.info(f"Checkpoint loaded from {path}")

    def to_cartesian(self, pred: dict):
        """Convert the predicted tensors to Cartesian tensors."""
        if self.M is not None:
            # Convert natural tensors to Cartesian tensors (n_configs, 3**rank)
            pred = self.converter.to_ordinary_tensor(pred, flatten_tensor_dim=True)
            # shape (n_configs, 6*6)
            pred = torch.matmul(pred, self.M)
        else:
            # Convert natural tensors to Cartesian tensors (n_configs, 3,3,...,3)
            pred = self.converter.to_ordinary_tensor(pred, flatten_tensor_dim=False)

        return pred

    # Abstract methods to be implemented by subclasses
    def compute_loss_nat(self, pred: dict, ref: dict):
        """Total loss for natural tensors."""
        raise NotImplementedError("Subclass must implement this method.")

    def compute_loss_cart(self, pred: torch.Tensor, ref: torch.Tensor):
        """Total loss for cartesian tensors."""
        raise NotImplementedError("Subclass must implement this method.")

    def compute_metrics(
        self,
        pred_nat: dict,
        ref_nat: dict,
        pred_cart: torch.Tensor = None,
        ref_cart: torch.Tensor = None,
        mode: str = "train",
    ):
        """Compute metrics for both natural and cartesian tensors."""
        metrics = self.compute_metrics_nat(pred_nat, ref_nat, mode)
        if self.target_mode in ["full", "voigt"]:
            metrics_cart = self.compute_metrics_cart(pred_cart, ref_cart, mode)
            metrics.update(metrics_cart)
        return metrics

    def compute_metrics_nat(self, pred: dict, ref: dict, mode: str):
        """Compute metrics for natural tensors."""
        raise NotImplementedError("Subclass must implement this method.")

    def compute_metrics_cart(self, pred: torch.Tensor, ref: torch.Tensor, mode: str):
        """Compute metrics for cartesian tensors."""
        raise NotImplementedError("Subclass must implement this method.")


class StructureTensorPyTorchTrainer(BasePyTorchTrainer):
    """PyTorch trainer for structure tensor prediction."""

    def compute_loss_nat(self, pred: dict, ref: dict):
        """Weighted MSE loss for natural tensors."""
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

    def compute_loss_cart(self, pred: torch.Tensor, ref: torch.Tensor):
        """MSE loss for cartesian tensors."""
        return {"train/loss_total": nn.functional.mse_loss(pred, ref, reduction="mean")}

    def compute_metrics_nat(self, pred: dict, ref: dict, mode: str):
        """MAE/MSE metrics for natural tensors."""
        metrics = {}
        for rank in pred:
            p = pred[rank]
            r = ref[str(rank)]

            # Update the metric object
            name = f"metrics_{mode}_{rank}"
            self.metrics[name](p, r)

            # Record the metric object to return
            metrics[f"{mode}/{self.metrics_type}_rank-{rank}"] = self.metrics[name]

        return metrics

    def compute_metrics_cart(self, pred: torch.Tensor, ref: torch.Tensor, mode: str):
        """MAE/MSE metrics for cartesian tensors."""
        name = f"metrics_{mode}_cartesian"
        self.metrics_cartesian[name](pred, ref)

        return {f"{mode}/{self.metrics_type}_cartesian": self.metrics_cartesian[name]}
