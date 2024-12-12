"""Atomic moment module."""

import torch
from torch import Tensor, nn

from carten.core.unit_vector import get_polyadics_from_vector

from .linear import LinearMap
from .mlp import MLP
from .product import TensorProduct
from .radial import RadialPart
from .scatter import scatter
from .utils import check_rank


class AtomicMoment(nn.Module):
    """
    Atomic Moments.

    Args:
        F: Number of features.
        L1: Max rank for the atomic features.
        L2: Max rank for the polyadics from the unit vectors.
        L3: Max rank for the tensor product of the atomic features and polyadics.
    """

    def __init__(
        self,
        F: int,
        L1: int,
        L2: int,
        L3: int | tuple[int, ...] | None,
        num_atom_types: int,
        num_average_neigh: float,
        max_chebyshev_degree: int = 8,
        radial_mlp_hidden_layers: list[int] | int = 2,
        r_cut: float = 5,
        envelope: int = 6,
    ):
        super().__init__()
        self.F = F
        self.L1 = L1
        self.L2 = L2
        self.L3 = check_rank(L1, L2, L3)
        self.num_atom_types = num_atom_types
        self.num_average_neigh = num_average_neigh
        self.max_chebyshev_degree = max_chebyshev_degree
        self.radial_mlp_hidden_layers = radial_mlp_hidden_layers
        self.r_cut = r_cut
        self.envelope = envelope

        # Radial part
        self.radial = RadialPart(
            F,
            num_atom_types,
            max_chebyshev_degree=max_chebyshev_degree,
            r_cut=r_cut,
            envelope=envelope,
        )

        # Tensor product
        self.tp = TensorProduct(F, L1, L2, L3, normalize="unity")

        # MLP on radial part, separate for each (l1, l2, l3)
        if isinstance(radial_mlp_hidden_layers, int):
            radial_mlp_hidden_layers = [F] * radial_mlp_hidden_layers

        self.radial_mlp = nn.ModuleDict()
        for paths in self.tp.paths.values():
            for p in paths:
                self.radial_mlp[str(p)] = MLP(
                    in_features=F,
                    out_features=F,
                    hidden_features=radial_mlp_hidden_layers,
                    out_activation=False,
                )

        # Linear combination of channels, separate for each l3
        self.linear_channel = nn.ModuleList([])
        for l3 in self.L3:
            if l3 == 0:
                bias = True
            else:
                bias = False
            self.linear_channel.append(LinearMap(F, F, bias))

    def forward(
        self,
        edge_vector: Tensor,
        edge_idx: Tensor,
        atom_type: Tensor,
        atom_feats: Tensor,
    ) -> Tensor:
        """

        Args:
            edge_vector: Edge vectors. Shape (n_edges, 3).
            edge_idx: Indices of center and neighbor atoms, that form the edges.
                Shape (2, n_edges). The first row is the center atom indices, and the
                second row is the neighbor atom indices.
            atom_type: Atom types. Shape (Na,), where Na is the number of atoms.
            atom_feats: shape (Na, F, T1), where Na is the number of atoms, F is the
                number of features, and T1 = (3**(L1+1)-1)//2 is the tensor dim.

        Returns:
            Updated atom feats. Shape (Na, F, T3), where T3 is the number tensor
            components determined by L3.
        """
        assert (
            atom_feats.shape[-1] == (3 ** (self.L1 + 1) - 1) // 2
        ), "Invalid atom feats shape."

        # Indices of center atoms (i) and neighbor atoms (j); (n_edges,)
        i_idx = edge_idx[0]
        j_idx = edge_idx[1]

        # Unit vectors of edges between center and neighbor atoms; (n_edges,)

        # Polyadics from unit vectors; (n_edges, T2)
        polyadics = get_polyadics_from_vector(
            nn.functional.normalize(edge_vector, p=2, dim=-1), self.L2
        )
        # (n_edges, F, T2)
        polyadics = polyadics.unsqueeze(-2).expand(
            polyadics.shape[0], self.F, polyadics.shape[1]
        )

        # Radial basis; (n_edges, F)
        fu = self.radial(
            torch.linalg.vector_norm(edge_vector, dim=-1),
            atom_type[i_idx],
            atom_type[j_idx],
        )

        # Radial params for each path
        R = {
            p: self.radial_mlp[str(p)](fu)
            for paths in self.tp.paths.values()
            for p in paths
        }

        # Tensor product of the features of center atoms and neighbor atoms
        # atom_feats[j_idx]: (n_edges, F, T1)
        # polyadics: (n_edges, F, T2)
        # R: (n_edges, F) for each path
        # product: (n_edges, F, T3)
        product = self.tp(atom_feats[j_idx], polyadics, R)

        # aggregate atoms j (src) to atom i (dst); (n_atoms, F, T3)
        M = scatter(product, i_idx, reduce="sum", dim=0) / self.num_average_neigh**0.5

        # TODO, we should move the mixing to the `Layer` module
        # TODO, is it possible to not do looping, maybe by constructing a kernel that
        #  combines all l3
        # Linear mix across channels
        start = 0
        for l3, fn in zip(self.L3, self.linear_channel):
            end = start + 3**l3
            # TODO, is the inplace OK? A: not for backprop
            M[..., start:end] = fn(M[..., start:end])
            start = end

        return M
