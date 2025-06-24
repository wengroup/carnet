"""CARTEN interatomic potential model."""

from torch import Tensor, nn

from .backbone import Backbone
from .readout import StructureScalar


class InteratomicPotential(nn.Module):
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
        atomic_moment_mode: str = "vanilla",
        tp_path_mode: str = "full",
        #
        layer_norm: bool = False,
        activation: str = None,
        last_layer_activation: bool = False,
        residual: bool = True,
        # optional layers
        use_linear_channel_input: bool = False,
        use_linear_channel_hyper: bool = False,
        use_linear_channel_residual: bool = True,
        # output
        output_mlp_hidden_layers: list[int] | int = 2,
        atomic_energy_shift: Tensor = None,
        atomic_energy_scale: Tensor = None,
        element_bias: bool = True,
    ):
        """
        Args:
            output_mlp_hidden_layers: Number of units in each hidden layer of the output
                MLP, which is applied to the last layer features before outputing.
                If a list of integers, each gives the number of units in the
                corresponding hidden layer. If an integer, this will be the number of
                hidden layers, and the number of units in each hidden layer is set to F,
                the channel dimension of the feature tensor.
            atomic_moment_mode: Architecture of the atomic moment: `vanilla`, `variant1`,
                or `variant2`.

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
            atomic_moment_mode=atomic_moment_mode,
            tp_path_mode=tp_path_mode,
            layer_norm=layer_norm,
            activation=activation,
            last_layer_activation=last_layer_activation,
            residual=residual,
            use_linear_channel_input=use_linear_channel_input,
            use_linear_channel_hyper=use_linear_channel_hyper,
            use_linear_channel_residual=use_linear_channel_residual,
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
        all_scalar_feats = self.backbone(
            edge_vector,
            edge_idx,
            atom_type,
            num_atoms,
            return_all=True,
            scalar_only=True,
        )

        # Compute the total energy
        energy = self.readout(all_scalar_feats, atom_type, num_atoms)

        return energy
