"""CARTEN backbone module that performs multiple iterations of feature updates."""

from typing import Optional

from torch import Tensor, nn

from carten.module.embedding import Embedding
from carten.module.layer import Layer


class Backbone(nn.Module):
    """
    Backbone block, consisting of multiple composed Layers.

    Args:
        F: Channel dimension.
        max_L: Max rank of the feature tensor. Natural tensors of rank 0, 1, ..., max_L
            will be included in the output feature tensor.
        num_atom_types: Number of atom types.
        r_cut: Cutoff radius for the radial basis functions.
        num_layers: Number of layers.
        num_average_neigh: average number of neighbors of the atoms
        max_out_L: Max rank for the output feature tensor in the last layer.
            If None, set to max_L.
        max_degree: Max correlation degree to construct the hyper moment tensor.
            If None, set to max_L.
        max_chebyshev_degree: max degree of the Chebyshev polynomial to use to construct
            the radial basis functions. The total number of chebyshev polynomials is
            `max_chebyshev_degree + 1`; +1 for the zeroth degree.
        radial_mlp_hidden_layers: if list of int, this gives the size of each hidden
            layer in the MLP that is applied to the radial basis functions. If int,
            this gives the number of hidden layers, and the size of each hidden
            layer is set to `max_u + 1`, the number of radial basis functions.
    """

    def __init__(
        self,
        F: int,
        max_L: int,
        num_atom_types: int,
        r_cut: float,
        num_layers: int,
        # TODO, do we need num_average_neigh?
        num_average_neigh: float,
        # angular
        max_out_L: int = None,
        max_degree: int = None,
        # radial
        max_chebyshev_degree: int = 8,
        radial_mlp_hidden_layers: list[int] | int = 2,
    ):
        super().__init__()
        self.F = F
        self.max_L = max_L
        self.num_atom_types = num_atom_types
        self.r_cut = r_cut
        self.num_layers = num_layers
        self.num_average_neigh = num_average_neigh

        self.max_out_L = max_L if max_out_L is None else max_out_L
        self.max_degree = max_L if max_degree is None else max_degree

        self.max_chebyshev_degree = max_chebyshev_degree
        self.radial_mlp_hidden_layers = radial_mlp_hidden_layers

        # Embed atom number as vectors
        self.atom_embedding = Embedding(num_atom_types, F)

        # First layer and last layer are a bit special.
        # For the first layer, the input features are only scalars (embedding of atom
        # types, L1=0), and we don't need to mix them.
        # For the last layer, we only need to select the interested output ranks
        # according to max_out_L (this can save some computation).
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                L1 = 0
                mix = False
                out_L = self.max_L
            elif i == num_layers - 1:
                L1 = max_L
                mix = True
                out_L = self.max_out_L
            else:
                L1 = max_L
                mix = True
                out_L = self.max_L

            self.layers.append(
                Layer(
                    F=self.F,
                    L1=L1,
                    L2=self.max_L,
                    L3=self.max_L,
                    num_atom_types=self.num_atom_types,
                    num_average_neigh=self.num_average_neigh,
                    max_chebyshev_degree=self.max_chebyshev_degree,
                    r_cut=self.r_cut,
                    radial_mlp_hidden_layers=self.radial_mlp_hidden_layers,
                    max_out_L=out_L,
                    max_degree=self.max_degree,
                    mix_atom_feats_across_channel=mix,
                )
            )

    def forward(
        self,
        edge_vector: Tensor,
        edge_idx: Tensor,
        atom_type: Tensor,
        num_atoms: Tensor,
        return_all_scalar_feats: bool = False,
    ) -> tuple[Tensor, list[Tensor]]:
        """
        Args:
            edge_vector:
            edge_idx:
            atom_type:
            num_atoms: 1D tensor of the number of atoms in each atomic configuration.
            return_all_scalar_feats: whether to output all scalar features from all
                layers.

        Returns:
            updated_feats: Updated atomic features after all layers.
                Shape (..., F, T'), where T' = ((max_out_L + 1)**2 -1))//2 is the total
                number of tensor components.
            scalar_feats: If `return_all_scalar_feats` is True, this is a list of scalar
                features from all layers. Each scalar feature is of shape (..., F, 1).
                If `return_all_scalar_feats` is False, this is an empty list.
        """
        # Embed atom number as scalar features of dim F; (n_atoms, F, 1)
        atom_feats = self.atom_embedding(atom_type).unsqueeze(-1)

        # TODO, for the first layer, because atom_feats is special (only scalars), we
        #  might be able to use a more constrained version of AtomicMoment in
        #  `Layer` to save some computation.
        scalar_feats = []
        for layer in self.layers:
            atom_feats = layer(edge_vector, edge_idx, atom_type, atom_feats)
            if return_all_scalar_feats:
                scalar_feats.append(atom_feats[..., 0:1])

        return atom_feats, scalar_feats
