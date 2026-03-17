from pathlib import Path

import torch
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes

from carnet.data.data import Config, get_edge_vec_single
from carnet.data.transform import ConsecutiveAtomType
from carnet.model.force_stress import (
    apply_strain_single_config,
    compute_forces,
    compute_forces_stress_single_config,
)
from carnet.model.ip import InteratomicPotential
from carnet.model.pl.pl_ip import InteratomicPotentialLitModule
from carnet.model.pl.utils import load_model


class CarnetCalculator(Calculator):
    """
    Calculator for the CAMP models.

    Args:
        model_path: Path to the trained model.
        use_ema_params: Whether to use the exponential moving average (EMA) parameters
            of the model. If False, the parameters at a single checkpoint are used.
            Typically, this should be "True".
        device: Device to run the model on. Options: "cpu", "cuda", "mps" etc.
        need_stress: This calculator can compute stress. But if you don't need stress,
            you can set this to `False` to save some computation time.
        override_atomic_numbers: Override the atomic numbers in the model with this list.
            If None, the atomic numbers in the model are used. Typically, this should be
            None. And None values are typically used for debugging purposes.
    """

    implemented_properties = ["energy", "forces", "stress"]

    def __init__(
        self,
        model_path: Path,
        use_ema_params: bool = True,
        device: str = "cpu",
        need_stress: bool = True,
        override_atomic_numbers: list[int] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.model_path = model_path
        self.use_ema_params = use_ema_params
        self.device = torch.device(device)
        self.need_stress = need_stress
        self.override_atomic_numbers = override_atomic_numbers

        lit_model = load_model(
            InteratomicPotentialLitModule,
            InteratomicPotential,
            model_path,
            map_location=device,
        )

        if use_ema_params:
            lit_model.ema.copy_params_from_ema_to_model()
        self.model = lit_model.model
        self.model.to(self.device)

        self.r_cut = lit_model.hparams["other_hparams"]["data"]["r_cut"]

        if override_atomic_numbers is not None:
            elements = override_atomic_numbers
        else:
            elements = lit_model.hparams["other_hparams"]["data"]["atomic_number"]

        self.supported_elements = set(elements)

        # Convert atomic number to atom type (consecutive integer starting from 0)
        self.transform = ConsecutiveAtomType(elements, device=self.device)
        self.atom_type = None

        self.results = {}

    def calculate(self, atoms=None, properties=["energy"], system_changes=all_changes):
        """
        Calculate properties.

        Args:
            atoms: ASE Atoms object
            properties: properties to be computed
            system_changes: system changes since last calculation, used by ASE internally
        """
        super().calculate(atoms, properties, system_changes)

        # not needed, since self.transform will check it
        # self._check_species(atoms)

        # prepare data
        data = Config.from_ase(atoms, r_cut=self.r_cut)
        data.to(self.device)
        data.pos.requires_grad_(True)

        # convert atomic numbers to consecutive integer code starting from 0, which
        # is what the model internally expects
        if self.atom_type is None:
            data = self.transform(data)
            self.atom_type = data.atom_type
        else:
            data.atom_type = self.atom_type

        has_cell = hasattr(data, "cell") and data.cell is not None

        if self.need_stress:
            if not has_cell:
                raise RuntimeError(
                    "Stress computation is requested but the cell is not present."
                )

            # apply strain and get strained positions and cell
            strain, strained_pos, strained_cell = apply_strain_single_config(
                data.pos, data.cell
            )
            data.pos = strained_pos
            data.cell = strained_cell

        cell = data.cell if has_cell else None
        edge_vector = get_edge_vec_single(
            data.pos, data.shift_vector, cell, data.edge_index
        )

        # Compute energy
        energy, _ = self.model(
            edge_vector,
            data.edge_index,
            data.atom_type,
            data.num_atoms,
            data.atomic_number,
        )

        # Compute forces (and stress)
        if self.need_stress:
            forces, stress = compute_forces_stress_single_config(
                energy, data.pos, data.cell, strain
            )
            stress = stress.detach().cpu().numpy()
        else:
            forces = compute_forces(energy, data.pos)
            stress = None

        self.results["energy"] = energy[0].item()
        self.results["forces"] = forces.detach().cpu().numpy()
        self.results["stress"] = stress

    def _check_species(self, atoms: Atoms):
        if set(atoms.get_atomic_numbers()) != self.supported_elements:
            raise ValueError(
                f"Supported atomic numbers should be in "
                f"{sorted(self.supported_elements)}, "
                f"got {sorted(set(atoms.get_atomic_numbers()))}."
            )
