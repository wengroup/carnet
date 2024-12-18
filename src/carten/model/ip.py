"""CARTEN interatomic potential model."""
from line_profiler import profile
from torch import Tensor, nn

from .backbone import Backbone
from .readout import StructureScalar


class InteratomicPotenital(nn.Module):
    """
    CARTEN interatomic potential model.
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
        max_degree: int = None,
        # radial
        max_chebyshev_degree: int = 8,
        radial_mlp_hidden_layers: list[int] | int = 2,
        # output
        output_mlp_hidden_layers: list[int] | int = 2,
        atomic_energy_shift: Tensor = None,
        atomic_energy_scale: Tensor = None,
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
            radial_mlp_hidden_layers=radial_mlp_hidden_layers,
        )

        self.readout = StructureScalar(
            num_layers=num_layers,
            in_features=F,
            hidden_features=output_mlp_hidden_layers,
            atomic_shift=atomic_energy_shift,
            atomic_scale=atomic_energy_scale,
        )

    @profile
    def forward(
        self,
        edge_vector: Tensor,
        edge_idx: Tensor,
        atom_type: Tensor,
        num_atoms: Tensor,
    ) -> Tensor:
        """
        Args:
            edge_vector: Edge vector tensor. Shape (n_edges, F).
            edge_idx: Edge index tensor. Shape (2, n_edges).
            atom_type: Atomic type of each atom. Shape (n_atoms,).
            num_atoms: Number of atoms in each atomic configuration. Shape (n_config,).

        Returns:
            Total energy of each atomic configuration. Shape (n_config,).
        """
        # Get the scalar feats from all layers
        _, all_scalar_feats = self.backbone(
            edge_vector, edge_idx, atom_type, num_atoms, return_all_scalar_feats=True
        )

        # Compute the total energy
        energy = self.readout(all_scalar_feats, atom_type, num_atoms)

        return energy
