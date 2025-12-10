"""Interatomic potential model."""

import torch
from torch import Tensor, nn

from carnet.module.scatter import scatter

from .backbone import Backbone
from .readout import StructureScalar
from .zbl import ZBLEnergy


class InteratomicPotential(nn.Module):
    """
    Interatomic potential model.
    """

    def __init__(
        self,
        F: int,
        max_L: int,
        num_atom_types: int,
        r_cut: float,
        num_layers: int,
        num_average_neigh: float,
        # angular
        max_degree: int = 3,
        # radial
        max_chebyshev_degree: int = 8,
        radial_part_type: int = 1,
        radial_mlp_hidden_layers: list[int] | int = 2,
        #
        tp_path_mode: str = "full",
        level: int = None,
        # optional layers
        use_linear_channel_input: bool = False,
        use_linear_channel_hyper: bool = False,
        use_linear_channel_residual: bool = True,
        use_atomic_dependent_weight: bool | str = True,
        residual: bool = True,
        # output
        output_mlp_hidden_layers: list[int] | int = 2,
        atomic_energy_shift: Tensor = None,
        atomic_energy_scale: Tensor = None,
        element_bias: bool = True,
        # zbl
        use_zbl: bool = False,
        zbl_trainable: bool = True,
    ):
        """
        Args:
            output_mlp_hidden_layers: Number of units in each hidden layer of the output
                MLP, which is applied to the last layer features before outputing.
                If a list of integers, each gives the number of units in the
                corresponding hidden layer. If an integer, this will be the number of
                hidden layers, and the number of units in each hidden layer is set to F,
                the channel dimension of the feature tensor.
        """
        super().__init__()

        self.backbone = Backbone(
            F=F,
            max_L=max_L,
            num_atom_types=num_atom_types,
            r_cut=r_cut,
            num_layers=num_layers,
            num_average_neigh=num_average_neigh,
            max_out_L=0,
            max_degree=max_degree,
            max_chebyshev_degree=max_chebyshev_degree,
            radial_part_type=radial_part_type,
            radial_mlp_hidden_layers=radial_mlp_hidden_layers,
            tp_path_mode=tp_path_mode,
            level=level,
            use_linear_channel_input=use_linear_channel_input,
            use_linear_channel_hyper=use_linear_channel_hyper,
            use_linear_channel_residual=use_linear_channel_residual,
            use_atomic_dependent_weight=use_atomic_dependent_weight,
            residual=residual,
        )

        self.readout = StructureScalar(
            num_layers=num_layers,
            in_features=F,
            hidden_features=output_mlp_hidden_layers,
            num_atom_types=num_atom_types,
            atomic_shift=atomic_energy_shift,
            atomic_scale=atomic_energy_scale,
            element_bias=element_bias,
        )

        if use_zbl:
            self.zbl = ZBLEnergy(trainable=zbl_trainable)
        else:
            self.register_buffer("zbl", None)

    def forward(
        self,
        edge_vector: Tensor,
        edge_idx: Tensor,
        atom_type: Tensor,
        num_atoms: Tensor,
        atomic_number: Tensor = None,
    ) -> Tensor:
        """
        Args:
            edge_vector: Edge vector tensor. Shape (n_edges, F).
            edge_idx: Edge index tensor. Shape (2, n_edges).
            atom_type: Atomic type of each atom. Shape (n_atoms,).
            num_atoms: Number of atoms in each atomic configuration. Shape (n_config,).
            atomic_number: Atomic number in each atomic configuration. Shape (n_atoms,).
                Note, this should be distinguished from `atom_type`. For example,
                for a system with three atoms, [H, H, O], the `atom_type` is [0, 0, 1]
                (H is type 0 and O is type 1), while the `atomic_number` is [1, 1, 8].

        Returns:
            Total energy of each atomic configuration. Shape (n_config,).
        """
        # Get the scalar feats from all layers
        all_scalar_feats = self.backbone(
            edge_vector,
            edge_idx,
            atom_type,
            num_atoms,
            return_all=True,
            scalar_only=True,
        )

        # Compute the total energy
        energy = self.readout(all_scalar_feats, atom_type, atomic_number, num_atoms)

        if self.zbl is not None:
            # ZBL energy of each atom
            zbl_atom = self.zbl(edge_vector, edge_idx, atomic_number)

            # ZBL energy of each configuration
            zbl_conf = scatter(
                zbl_atom, torch.repeat_interleave(num_atoms), reduce="sum", dim=0
            )

            energy = energy + zbl_conf

        return energy
