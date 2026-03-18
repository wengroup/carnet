from __future__ import annotations

from pathlib import Path

import torch

from carnet.ext.lammps.mliap import LAMMPS_MLIAP_CarNet
from carnet.model.ip import InteratomicPotential
from carnet.model.pl.pl_ip import InteratomicPotentialLitModule
from carnet.model.pl.utils import load_model


def export_model_for_lammps_mliap(
    model_path: str | Path, use_ema: bool = True, map_location: str = "cpu"
):
    """
    Load a CarNet model from a checkpoint and prepare it for LAMMPS ML-IAP.

    Args:
        model_path: Path to the trained CarNet model checkpoint.
        use_ema: Whether to use the EMA parameters of the model.
        map_location: Device to load the model to.

    Returns:
        The InteratomicPotential model instance with metadata attached.
    """
    model_path = Path(model_path)

    # Load LitModule
    lit_model = load_model(
        InteratomicPotentialLitModule,
        InteratomicPotential,
        model_path,
        map_location=map_location,
    )

    # Preprocessing: handle EMA
    if use_ema:
        lit_model.ema.copy_params_from_ema_to_model()

    model = lit_model.model

    # Extract metadata from LitModule hyperparameters
    hparams = lit_model.hparams["other_hparams"]["data"]
    model.atomic_numbers = hparams["atomic_number"]
    model.r_cut = hparams["r_cut"]

    mliap_model = LAMMPS_MLIAP_CarNet(model)

    torch.save(mliap_model, model_path.stem + "-lammps_mliap.pt")


if __name__ == "__main__":
    # Path to a CarNet model checkpoint
    path = "/Users/mjwen/Packages/carnet/scripts/last_epoch.ckpt"
    export_model_for_lammps_mliap(path)
