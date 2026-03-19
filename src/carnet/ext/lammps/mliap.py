from __future__ import annotations

import logging
import os
import sys
import time
from contextlib import contextmanager
from typing import Any, Dict, Tuple

import torch
from ase.data import chemical_symbols

from carnet._dtype import DTYPE_INT

try:
    from lammps.mliap.mliap_unified_abc import MLIAPUnified
except ImportError:

    class MLIAPUnified:
        def __init__(self, *args, **kwargs):
            pass


from carnet.data.transform import ConsecutiveAtomType
from carnet.model.ip import InteratomicPotential


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
        self.dtype = getattr(torch, dtype)
        self.device = device

        # Set standard MLIAP Unified fields
        self.element_types = [chemical_symbols[s] for s in model.atomic_numbers]
        self.rcutfac = 0.5 * model.r_cut  # 0.5: a 2 is multiplied in mliap_unified.cpp
        self.ndescriptors = 1
        self.nparams = 1

        # Mapping atomic numbers to contiguous indices
        transform = ConsecutiveAtomType(model.atomic_numbers)
        self.mapping = transform.mapping

        # self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

        self.config = CarNetLammpsConfig()
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
        self.model = self.model.to(self.device)
        self.mapping = self.mapping.to(self.device)

        self.initialized = True
        logging.info(f"LAMMPS_MLIAP_CarNet model loaded on {self.device}")

    def compute_forces(self, data):
        """Compute forces and per-atom energies for LAMMPS."""
        nlocal = data.nlocal
        ntotal = data.ntotal
        nghost = ntotal - nlocal
        npairs = data.npairs

        if not self.initialized:
            self._initialize_device(data)

        self.step += 1
        self._manage_profiling()

        if nlocal == 0 or npairs <= 1:
            return

        with timer("total_step", enabled=self.config.debug_time):
            with timer("prepare_batch", enabled=self.config.debug_time):
                batch = self._prepare_batch(data, ntotal, nlocal, nghost)

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
        batch: torch.Tensor,
        lammps_natoms: Tuple[int, int],
        lammps_class: Any,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute energies and per-pair forces."""
        edge_vector.requires_grad_(True)

        _, e_atom = self.model(
            edge_vector,
            edge_index,
            atom_type,
            num_atoms,
            atomic_number,
            batch,
            lammps_natoms=lammps_natoms,
            lammps_class=lammps_class,
        )

        # force = -dE/dR, but here we do not apply -1 to match LAMMPS sign convention
        pair_forces = torch.autograd.grad(
            outputs=e_atom.sum(),
            inputs=edge_vector,
            retain_graph=False,
            create_graph=False,
            allow_unused=True,
        )[0]

        if pair_forces is None:
            pair_forces = torch.zeros_like(edge_vector)

        return e_atom, pair_forces

    def _prepare_batch(self, data, ntotal, nlocal, nghost) -> Dict[str, torch.Tensor]:
        """Prepare the input batch for the CarNet model."""

        atomic_number = torch.as_tensor(data.elems, dtype=DTYPE_INT, device=self.device)
        atomic_number += 1  # +1 to convert zero-based lammps value to one-based
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
            "batch": torch.zeros(ntotal, dtype=DTYPE_INT, device=self.device),
            "lammps_natoms": (nlocal, nghost),
            "lammps_class": data,
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
