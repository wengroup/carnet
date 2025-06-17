"""CARTEN backbone module that performs multiple iterations of feature updates."""

from torch import Tensor, nn

from carten.module.embedding import Embedding
from carten.module.layer import Layer


class Backbone(nn.Module):
    """
    Backbone block, consisting of multiple composed Layers.

    Args:
        F: Channel dimension.
        max_L: Max rank of the feature tensor. Natural tensors of rank 0, 1, ..., max_L
            will be used in the tensor product.
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
        atomic_moment_mode: Architecture of the atomic moment: `vanilla`, `variant1`,
            or `variant2`.
        activation: Nonlinear activation function to apply after each layer.
        last_layer_activation: Whether to apply the activation function after the last
            layer.
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
        atomic_moment_mode: str = "vanilla",
        # activation
        activation: str = None,
        last_layer_activation: bool = False,
        # residual connection
        residual: bool = True,
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

        self.atomic_moment_mode = atomic_moment_mode

        # Embed atom number as vectors
        self.atom_embedding = Embedding(num_atom_types, F)

        # First layer and last layer are a bit special.
        # For the first layer, the input features are only scalars (embedding of atom
        # types, L1=0).
        # For the last layer, we only need to select the interested output ranks
        # according to max_out_L (this can save some computation).
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                L1 = 0
                out_L = self.max_L
                act = activation
            elif i == num_layers - 1:
                # TODO, this can be further optimized by replacing `out_L` with explicit
                #  degrees of the natural tensors. For example, for elastic tensor,
                #  we only need ranks 0, 2, and 4. The current implementation that uses
                #  `max_out_L=4` will compute the features of ranks 0 to 4.
                L1 = max_L
                out_L = self.max_out_L
                act = activation if last_layer_activation else None
            else:
                L1 = max_L
                out_L = self.max_L
                act = activation

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
                    atomic_moment_mode=self.atomic_moment_mode,
                    activation=act,
                    residual=residual,
                )
            )

    def forward(
        self,
        edge_vector: Tensor,
        edge_idx: Tensor,
        atom_type: Tensor,
        num_atoms: Tensor,
        return_all: bool = False,
        scalar_only: bool = False,
    ) -> list[Tensor]:
        """
        Args:
            edge_vector:
            edge_idx:
            atom_type:
            num_atoms: 1D tensor of the number of atoms in each atomic configuration.
            return_all: Whether to return the atom features from all layers.
            scalar_only: Whether to return only the scalar features.


        Returns:
            atom_feats: Updated atomic features. The output is a list of tensors.
                `return_all` determines the size of the list. If `False`, the list
                consists of a single tensor, which is the updated features from the last
                layer. If `True`, the list contains the updated features from all
                layers. `scalar_only` determines the shape of the tensors in the list.
                If `True`, the tensors are of shape (..., F, 1).
                if `False`, the tensors are of shape (..., F, T'), where
                T' = (3**(max_out_L + 1) -1))//2 is the total number of tensor
                components. Note, max_out_L can be smaller than max_L, so we need to do
                the selection.

        """

        # Embed atom number as scalar features of dim F; (n_atoms, F, 1)
        atom_feats = self.atom_embedding(atom_type).unsqueeze(-1)

        # TODO, for the first layer, because atom_feats is special (only scalars), we
        #  might be able to use a more constrained version of AtomicMoment in
        #  `Layer` to save some computation.
        output = []
        for i, layer in enumerate(self.layers):

            atom_feats = layer(edge_vector, edge_idx, atom_type, atom_feats)

            # Gather output
            if return_all or i == self.num_layers - 1:
                if scalar_only:
                    x = atom_feats[..., 0:1]
                else:
                    x = atom_feats[..., : (3 ** (self.max_out_L + 1) - 1) // 2]

                output.append(x)

        return output
