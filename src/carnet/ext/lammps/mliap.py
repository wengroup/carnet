from __future__ import annotations

import logging
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Tuple

import torch

from carnet._dtype import DTYPE_INT

try:
    from lammps.mliap.mliap_unified_abc import MLIAPUnified
except ImportError:

    class MLIAPUnified:
        def __init__(self, *args, **kwargs):
            pass


# CarNet specific imports
from carnet.data.transform import ConsecutiveAtomType
from carnet.model.ip import InteratomicPotential
from carnet.model.pl.pl_ip import InteratomicPotentialLitModule
from carnet.model.pl.utils import load_model


class CarNetLammpsConfig:
    """Configuration settings for CarNet-LAMMPS integration."""

    def __init__(self):
        self.debug_time = self._get_env_bool("CARNET_TIME", False)
        self.debug_profile = self._get_env_bool("CARNET_PROFILE", False)
        self.profile_start_step = int(os.environ.get("CARNET_PROFILE_START", "5"))
        self.profile_end_step = int(os.environ.get("CARNET_PROFILE_END", "10"))
        self.allow_cpu = self._get_env_bool("CARNET_ALLOW_CPU", False)
        self.force_cpu = self._get_env_bool("CARNET_FORCE_CPU", False)

    @staticmethod
    def _get_env_bool(var_name: str, default: bool) -> bool:
        return os.environ.get(var_name, str(default)).lower() in (
            "true",
            "1",
            "t",
            "yes",
        )


@contextmanager
def timer(name: str, enabled: bool = True):
    """Context manager for timing code blocks."""
    if not enabled:
        yield
        return

    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        logging.info(f"Timer - {name}: {elapsed*1000:.3f} ms")


class LAMMPS_MLIAP_CarNet(MLIAPUnified):
    """
    LAMMPS ML-IAP Unified interface for CarNet.

    Example LAMMPS input (when using a Python script to set up LAMMPS):
    ```python
    from carnet.ext.lammps.mliap import load_model_for_lammps, LAMMPS_MLIAP_CarNet
    model = load_model_for_lammps("model.ckpt", use_ema=True)
    mliap = LAMMPS_MLIAP_CarNet(model, device="cuda")
    # pass mliap to lammps
    ```
    """

    def __init__(
        self, model: InteratomicPotential, dtype: str = "float32", device: str = "cpu"
    ):
        """
        Initialize the CarNet ML-IAP interface.

        Args:
            model: An instance of InteratomicPotential (prepared by load_model_for_lammps).
            device: Device to run the model on (e.g., 'cpu', 'cuda').
        """
        super().__init__()
        self.model = model
        self.device = device
        self.dtype = getattr(torch, dtype)

        self.config = CarNetLammpsConfig()

        self.atomic_numbers = self.model.atomic_numbers
        r_cut = self.model.r_cut

        # Set standard MLIAP Unified fields
        self.rcutfac = 0.5 * r_cut  # 0.5 since lammps multiplies 2 in mliap_unified.cpp
        self.ndescriptors = 1
        self.nparams = 1

        self.initialized = False
        self.step = 0

        logging.info("LAMMPS_MLIAP_CarNet initialized")

    def _initialize_device(self, data):
        """Initialize the model on the correct device."""
        using_kokkos = "kokkos" in data.__class__.__module__.lower()

        if using_kokkos and not self.config.force_cpu:
            device = torch.as_tensor(data.elems).device
            if device.type == "cpu" and not self.config.allow_cpu:
                raise ValueError(
                    "GPU requested but tensor is on CPU. Set CARNET_ALLOW_CPU=true to allow CPU computation."
                )
        else:
            device = torch.device(self.device)

        self.device = device

        # Prepare model
        self.model = self.model.to(self.device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

        # Transform for mapping atomic numbers to contiguous indices
        transform = ConsecutiveAtomType(self.atomic_numbers, device=self.device)
        self.mapping = transform.mapping

        # Clear temporary attributes
        self.device = None

        self.initialized = True
        logging.info(f"LAMMPS_MLIAP_CarNet model loaded on {self.device}")

    def compute_forces(self, data):
        """Compute forces and per-atom energies for LAMMPS."""
        nlocal = data.nlocal
        ntotal = data.ntotal

        if not self.initialized:
            self._initialize_device(data)

        self.step += 1
        self._manage_profiling()

        if nlocal == 0:
            return

        with timer("total_step", enabled=self.config.debug_time):
            with timer("prepare_batch", enabled=self.config.debug_time):
                batch = self._prepare_batch(data, ntotal)

            with timer("model_forward", enabled=self.config.debug_time):
                atom_energies, pair_forces = self._model_forward(**batch)

                if self.device.type != "cpu":
                    torch.cuda.synchronize()

            with timer("update_lammps", enabled=self.config.debug_time):
                self._update_lammps_data(data, atom_energies, pair_forces, nlocal)

    def _model_forward(
        self,
        edge_vector: torch.Tensor,
        edge_index: torch.Tensor,
        atom_type: torch.Tensor,
        num_atoms: torch.Tensor,
        atomic_number: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute energies and per-pair forces."""
        # Ensure edge_vector requires grad for force computation
        edge_vector.requires_grad_(True)

        # 1. Forward pass to get per-atom energy
        _, e_atom = self.model(
            edge_vector,
            edge_index,
            atom_type,
            num_atoms,
            atomic_number,
        )

        # 2. Backward pass for pairwise forces
        # pair_forces[e] = - d(sum(e_atom)) / d(rij[e])
        pair_forces = -torch.autograd.grad(
            e_atom.sum(), edge_vector, retain_graph=False
        )[0]

        return e_atom, pair_forces

    def _prepare_batch(self, data, ntotal) -> Dict[str, torch.Tensor]:
        """Prepare the input batch for the CarNet model."""
        atomic_number = torch.as_tensor(data.elems, dtype=DTYPE_INT, device=self.device)
        atom_type = self.mapping[atomic_number]

        return {
            "edge_vector": torch.as_tensor(
                data.rij, dtype=self.dtype, device=self.device
            ),
            "edge_index": torch.stack(
                [
                    torch.as_tensor(data.pair_i, dtype=DTYPE_INT, device=self.device),
                    torch.as_tensor(data.pair_j, dtype=DTYPE_INT, device=self.device),
                ],
                dim=0,
            ),
            "atom_type": atom_type,
            "num_atoms": torch.tensor([ntotal], dtype=DTYPE_INT, device=self.device),
            "atomic_number": atomic_number,
        }

    def _update_lammps_data(self, data, atom_energies, pair_forces, nlocal):
        """Update LAMMPS data structures with computed energies and forces."""
        # Per-atom energies (only for local atoms)
        eatoms = torch.as_tensor(data.eatoms)
        eatoms.copy_(atom_energies[:nlocal].detach())

        # Total energy of local atoms
        data.energy = atom_energies[:nlocal].sum().item()

        # Update pairwise forces
        if self.dtype == torch.float32:
            pair_forces = pair_forces.double()

        # TODO, should we use update_pair_forces if on CPU?
        data.update_pair_forces_gpu(pair_forces)

    def _manage_profiling(self):
        if not self.config.debug_profile:
            return

        if self.step == self.config.profile_start_step:
            logging.info(f"Starting CUDA profiler at step {self.step}")
            torch.cuda.profiler.start()

        if self.step == self.config.profile_end_step:
            logging.info(f"Stopping CUDA profiler at step {self.step}")
            torch.cuda.profiler.stop()
            logging.info("Profiling complete. Exiting.")
            sys.exit()

    def compute_descriptors(self, data):
        pass

    def compute_gradients(self, data):
        pass


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
