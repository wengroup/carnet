"""Carten model to predict tensorial properties of materials and molecules."""
from torch import Tensor, nn

from .backbone import Backbone
from .readout import AtomicTensor, StructureTensor


class AtomicTensorModel(nn.Module):
    """
    CARTEN model to predict a tensorial property for each atom in a system, such as
    NMR shielding tensors.
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
        atomic_moment_mode: str = "vanilla",
        output_signature: dict[int, int] = None,
        output_from_all_layers: bool = False,
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
            output_signature: A dictionary {l: n_l} that specifies the natural tensor
                components to output for each atomic configuration. The key `l` gives
                the rank of the natural tensor, and the value `n_l` gives the number of
                rank-l natural tensor to output. For example, a dielectric tensor is a
                rank-2 tensor, which can be decomposed as 1 rank-0, 1 rank-1, and
                1 rank-2 natural tensors. To model the dielectric tensor, the
                output_signature should be {0: 1, 1: 1, 2: 1}. As another example,
                the elastic tensor is a rank-4 tensor, which can be decomposed as
                2 rank-0, 2 rank-2, and 1 rank-4 natural tensors. To model the elastic
                tensor, the output_signature should be {0: 2, 2: 2, 4: 1}.
            output_from_all_layers: If True, the output is constructed from the
                atom features of all layers. If False, the output is constructed from
                the atom features of the last layer only.
        """
        super().__init__()

        self.output_from_all_layers = output_from_all_layers
        if output_from_all_layers:
            num_atom_feats = num_layers
        else:
            num_atom_feats = 1

        self.backbone = Backbone(
            F=F,
            max_L=max_L,
            num_atom_types=num_atom_types,
            r_cut=r_cut,
            num_layers=num_layers,
            num_average_neigh=num_average_neigh,
            max_out_L=max(output_signature.keys()),
            max_degree=max_degree,
            max_chebyshev_degree=max_chebyshev_degree,
            radial_mlp_hidden_layers=radial_mlp_hidden_layers,
            atomic_moment_mode=atomic_moment_mode,
        )

        self.readout = AtomicTensor(
            num_layers=num_layers,
            in_features=F,
            hidden_features=output_mlp_hidden_layers,
            output_signature=output_signature,
            num_atom_feats=num_atom_feats,
        )

    def forward(
        self,
        edge_vector: Tensor,
        edge_idx: Tensor,
        atom_type: Tensor,
        num_atoms: Tensor,
    ) -> Tensor:
        # Get the atom feats
        all_atom_feats = self.backbone(
            edge_vector,
            edge_idx,
            atom_type,
            num_atoms,
            return_all=self.output_from_all_layers,
            scalar_only=False,
        )

        # Compute the atomic tensor on each atom
        output = self.readout(all_atom_feats, atom_type, num_atoms)

        return output


class StructureTensorModel(nn.Module):
    """
    CARTEN model to predict a tensorial property for a material or molecular structure,
    such as dielectric and elastic tensors.
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
        atomic_moment_mode: str = "vanilla",
        output_signature: dict[int, int] = None,
        output_from_all_layers: bool = False,
    ):
        super().__init__()

        self.output_from_all_layers = output_from_all_layers
        if output_from_all_layers:
            num_atom_feats = num_layers
        else:
            num_atom_feats = 1

        self.backbone = Backbone(
            F=F,
            max_L=max_L,
            num_atom_types=num_atom_types,
            r_cut=r_cut,
            num_layers=num_layers,
            num_average_neigh=num_average_neigh,
            max_out_L=max(output_signature.keys()),
            max_degree=max_degree,
            max_chebyshev_degree=max_chebyshev_degree,
            radial_mlp_hidden_layers=radial_mlp_hidden_layers,
            atomic_moment_mode=atomic_moment_mode,
        )

        self.readout = StructureTensor(
            num_layers=num_layers,
            in_features=F,
            hidden_features=output_mlp_hidden_layers,
            output_signature=output_signature,
            num_atom_feats=num_atom_feats,
        )

    def forward(
        self,
        edge_vector: Tensor,
        edge_idx: Tensor,
        atom_type: Tensor,
        num_atoms: Tensor,
    ) -> Tensor:
        # Get the atom feats
        all_atom_feats = self.backbone(
            edge_vector,
            edge_idx,
            atom_type,
            num_atoms,
            return_all=self.output_from_all_layers,
            scalar_only=False,
        )

        # Compute the structure tensor of each configuration
        output = self.readout(all_atom_feats, atom_type, num_atoms)

        return output
