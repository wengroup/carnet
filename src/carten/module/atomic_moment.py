"""Atomic moment module."""

import torch
from torch import Tensor, nn

from carten.core.unit_vector import get_polyadics_from_vector

from .mlp import MLP
from .product import TensorProduct
from .radial import RadialPart
from .scatter import scatter
from .utils import check_rank


class AtomicMoment(nn.Module):
    """
    Atomic Moments.

    Args:
        F: Channel dimension.
        L1: Max rank for the atomic features.
        L2: Max rank for the polyadics from the unit vectors.
        L3: Max rank for the tensor product of the atomic features and polyadics.
    """

    def __init__(
        self,
        F: int,
        L1: int,
        L2: int,
        L3: int | list[int] | None,
        num_atom_types: int,
        num_average_neigh: float,
        max_chebyshev_degree: int = 8,
        radial_mlp_hidden_layers: list[int] | int = 2,
        r_cut: float = 5,
        envelope: int = 6,
        mode: str = None,  # dummy argument to give the same signature as AtomicMoment2
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
                p = str(p)
                self.radial_mlp[p] = MLP(
                    in_features=F,
                    out_features=F,
                    hidden_features=radial_mlp_hidden_layers,
                    out_activation=False,
                )

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

        # TODO, this can be computed once and reused across all layers
        # Polyadics of unit vectors; (n_edges, T2)
        polyadics = get_polyadics_from_vector(
            nn.functional.normalize(edge_vector, p=2.0, dim=-1), self.L2
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

        # TODO, the radial_mlp might be batched
        #  put all weights as a larger one
        # Radial params for each path
        R = {p: fn(fu) for p, fn in self.radial_mlp.items()}

        # Tensor product of the features of center atoms and neighbor atoms
        # atom_feats[j_idx]: (n_edges, F, T1)
        # polyadics: (n_edges, F, T2)
        # R: (n_edges, F) for each path
        # product: (n_edges, F, T3)
        product = self.tp(atom_feats[j_idx], polyadics, R)

        # aggregate atoms j (src) to atom i (dst); (n_atoms, F, T3)
        M = scatter(product, i_idx, reduce="sum", dim=0) / self.num_average_neigh**0.5

        return M


# TODO, this is not used, delete
class AtomicMoment2(nn.Module):
    """
    Atomic variant 1 and variant 2, where the tensor product is done after the sum
    of neighboring atom features and polyadics.

    See carten docs.

    Args:
        F: Channel dimension.
        L1: Max rank for the atomic features.
        L2: Max rank for the polyadics from the unit vectors.
        L3: Max rank for the tensor product of the atomic features and polyadics.
        mode: `variant1` or `variant2`. `varinat1` does not use features of neighboring
            atoms in the tensor product, while `variant2` uses scalar features of
            neighboring atoms.
    """

    def __init__(
        self,
        F: int,
        L1: int,
        L2: int,
        L3: int | list[int] | None,
        num_atom_types: int,
        num_average_neigh: float,
        max_chebyshev_degree: int = 8,
        radial_mlp_hidden_layers: list[int] | int = 2,
        r_cut: float = 5,
        envelope: int = 6,
        mode: str = "variant1",
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
        self.mode = mode

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

        # MLP on radial part, only depends on the polyadics (l2)
        if isinstance(radial_mlp_hidden_layers, int):
            radial_mlp_hidden_layers = [F] * radial_mlp_hidden_layers
        self.radial_mlp = nn.ModuleList(
            [
                MLP(
                    in_features=F,
                    out_features=F,
                    hidden_features=radial_mlp_hidden_layers,
                    out_activation=False,
                )
                for _ in range(L2 + 1)
            ]
        )

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

        # Polyadics of unit vectors; (n_edges, T2)
        polyadics = get_polyadics_from_vector(
            nn.functional.normalize(edge_vector, p=2.0, dim=-1), self.L2
        )

        # Radial basis; (n_edges, F)
        fu = self.radial(
            torch.linalg.vector_norm(edge_vector, dim=-1),
            atom_type[i_idx],
            atom_type[j_idx],
        )

        # TODO, the radial_mlp might be batched
        #  put all weights as a larger one
        # Radial params for each l2; each element in the list is of shape (n_edges, F)
        R = [fn(fu) for fn in self.radial_mlp]

        # Concatenate along tensor dim; (n_edges, F, T2)
        R_cat = torch.cat(
            [
                r.unsqueeze(-1).expand(-1, -1, 3**l2)
                for r, l2 in zip(R, range(self.L2 + 1))
            ],
            dim=-1,
        )

        # Evaluate RD or RhD; (n_edges, F, T2)
        # R_cat: (n_edges, F, T2)
        # polyadics: (n_edges, T2)
        # atom_feats[j_idx][:1]: (n_edges, F, 1)
        if self.mode == "variant1":
            rp = R_cat * polyadics.unsqueeze(-2)
        elif self.mode == "variant2":
            rp = R_cat * atom_feats[j_idx][:1] * polyadics.unsqueeze(-2)
        else:
            raise ValueError(f"Invalid mode: {self.mode}")

        # aggregate atoms j (neighbors) to atom i (center); (n_atoms, F, T2)
        rp_sum = scatter(rp, i_idx, reduce="sum", dim=0) / self.num_average_neigh**0.5

        # Tensor product of the features of atom i and the sum of rp of atoms j
        # atom_feats[j_idx]: (atoms, F, T1)
        # rp_sum: (n_atoms, F, T2)
        # M: (n_atoms, F, T3)
        M = self.tp(atom_feats, rp_sum)

        return M
