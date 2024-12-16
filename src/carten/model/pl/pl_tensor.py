"""Lightning model for predicting tensorial properties of materials and molecules.

This is supposed to be used with the `carten.model.tensor.StructureTensorModel`
and `carten.model.tensor.AtomicTensorModel` models.
"""
from typing import Any

from lightning import LightningModule
from lightning.pytorch.cli import instantiate_class
from lightning.pytorch.utilities import grad_norm
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torch import nn


class LitModelStructureTensor(LightningModule):
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
        super().__init__()


class LitModelAtomicTensor(LightningModule):
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
        super().__init__()
