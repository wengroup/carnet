"""Atomic moment module."""

from typing import Optional

import torch
from torch import Tensor, nn

from carnet.core.unit_vector import Polyadics

from .mlp import MLP
from .product import TensorProduct
from .radial import RadialPart1, RadialPart2, RadialPart3, RadialPart4
from .scatter import scatter
from .utils import check_rank


class AtomicMoment(nn.Module):
    """
    Atomic Moments.

    For each path (l1, l2, l3) in the tensor product of atomic features and
    polyadics of unit vectors, an independent weight is used to combine the
    results. For efficiency, the weights for all paths are computed using a
    single batched MLP.

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
        radial_part_type: int = 1,
        radial_mlp_hidden_layers: list[int] | int = 2,
        r_cut: float = 5,
        envelope: int = 6,
        tp_path_mode: str = "lite",
        tp_path_polar_only: bool = False,
        level: int = None,
    ):
        super().__init__()
        self.F = F
        self.L1 = L1
        self.L2 = L2
        self.L3 = check_rank(L1, L2, L3)
        self.num_atom_types = num_atom_types
        self.num_average_neigh = num_average_neigh
        self.max_chebyshev_degree = max_chebyshev_degree
        self.radial_part_type = radial_part_type
        self.radial_mlp_hidden_layers = radial_mlp_hidden_layers
        self.r_cut = r_cut
        self.envelope = envelope
        self.tp_path_mode = tp_path_mode
        self.level = level

        # Polyadics
        self.polyadics_module = Polyadics(L2, normalize="unity")

        # Tensor product
        self.tp = TensorProduct(
            F,
            L1,
            L2,
            L3,
            normalize="unity",
            path_mode=tp_path_mode,
            path_polar_only=tp_path_polar_only,
            level=level,
            for_atomic_moment=True,
        )

        if radial_part_type == 1:
            RadialPartCalss = RadialPart1
        elif radial_part_type == 2:
            RadialPartCalss = RadialPart2
        elif radial_part_type == 3:
            RadialPartCalss = RadialPart3
        elif radial_part_type == 4:
            RadialPartCalss = RadialPart4
        else:
            raise ValueError(f"Invalid radial_part_type: {self.radial_part_type}")

        self.radial = RadialPartCalss(
            F, num_atom_types, max_chebyshev_degree, r_cut, envelope
        )
        radial_output_dim = self.radial.output_dim

        # MLP on radial part.
        # Although each path (l1, l2, l3) has its own independent weights, we
        # combine them into a single MLP for efficiency.
        if isinstance(radial_mlp_hidden_layers, int):
            radial_mlp_hidden_layers = [F] * radial_mlp_hidden_layers
        self.all_paths = [str(p) for paths in self.tp.paths.values() for p in paths]
        self.radial_mlp = MLP(
            in_features=radial_output_dim,
            out_features=F * len(self.all_paths),
            hidden_features=radial_mlp_hidden_layers,
            out_activation=False,
        )

    def forward(
        self,
        edge_vector: Tensor,
        edge_idx: Tensor,
        atom_type: Tensor,
        atom_feats: Tensor,
        polyadics: Optional[Tensor] = None,
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
            polyadics: Optional precomputed polyadics of unit vectors.
                Shape (n_edges, T2). If None, they are computed in this module.

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
        if polyadics is None:
            polyadics = self.polyadics_module(
                nn.functional.normalize(edge_vector, p=2.0, dim=-1)
            )

        # Radial basis; (n_edges, output_dim)
        fu = self.radial(
            torch.linalg.vector_norm(edge_vector, dim=-1),
            atom_type[i_idx],
            atom_type[j_idx],
        )

        # Radial params for all paths; (n_edges, F * n_paths)
        R_all = self.radial_mlp(fu)

        # Reshape to (n_edges, n_paths, F) to match TensorProduct interface.
        R_all = R_all.view(-1, len(self.all_paths), self.F)

        # Tensor product of the features of center atoms and neighbor atoms
        # atom_feats[j_idx]: (n_edges, F, T1)
        # polyadics: (n_edges, T2)
        # R_all: (n_edges, n_paths, F)
        # product: (n_edges, F, T3)
        product = self.tp(atom_feats[j_idx], polyadics, R_all)

        # aggregate atoms j (src) to atom i (dst); (n_atoms, F, T3)
        M = scatter(product, i_idx, reduce="sum", dim=0) / self.num_average_neigh**0.5

        return M


#
# # TODO, this is not used, delete
# class AtomicMoment2(nn.Module):
#     """
#     Atomic variant 1 and variant 2, where the tensor product is done after the sum
#     of neighboring atom features and polyadics.
#
#     See carnet docs.
#
#     Args:
#         F: Channel dimension.
#         L1: Max rank for the atomic features.
#         L2: Max rank for the polyadics from the unit vectors.
#         L3: Max rank for the tensor product of the atomic features and polyadics.
#         mode: `variant1` or `variant2`. `varinat1` does not use features of neighboring
#             atoms in the tensor product, while `variant2` uses scalar features of
#             neighboring atoms.
#     """
#
#     def __init__(
#         self,
#         F: int,
#         L1: int,
#         L2: int,
#         L3: int | list[int] | None,
#         num_atom_types: int,
#         num_average_neigh: float,
#         max_chebyshev_degree: int = 8,
#         radial_mlp_hidden_layers: list[int] | int = 2,
#         r_cut: float = 5,
#         envelope: int = 6,
#         mode: str = "variant1",
#     ):
#         super().__init__()
#         self.F = F
#         self.L1 = L1
#         self.L2 = L2
#         self.L3 = check_rank(L1, L2, L3)
#         self.num_atom_types = num_atom_types
#         self.num_average_neigh = num_average_neigh
#         self.max_chebyshev_degree = max_chebyshev_degree
#         self.radial_mlp_hidden_layers = radial_mlp_hidden_layers
#         self.r_cut = r_cut
#         self.envelope = envelope
#         self.mode = mode
#
#         # Radial part
#         self.radial = RadialPart(
#             F,
#             num_atom_types,
#             max_chebyshev_degree=max_chebyshev_degree,
#             r_cut=r_cut,
#             envelope=envelope,
#         )
#
#         # Tensor product
#         self.tp = TensorProduct(F, L1, L2, L3, normalize="unity")
#
#         # MLP on radial part, only depends on the polyadics (l2)
#         if isinstance(radial_mlp_hidden_layers, int):
#             radial_mlp_hidden_layers = [F] * radial_mlp_hidden_layers
#         self.radial_mlp = nn.ModuleList(
#             [
#                 MLP(
#                     in_features=F,
#                     out_features=F,
#                     hidden_features=radial_mlp_hidden_layers,
#                     out_activation=False,
#                 )
#                 for _ in range(L2 + 1)
#             ]
#         )
#
#     def forward(
#         self,
#         edge_vector: Tensor,
#         edge_idx: Tensor,
#         atom_type: Tensor,
#         atom_feats: Tensor,
#     ) -> Tensor:
#         """
#
#         Args:
#             edge_vector: Edge vectors. Shape (n_edges, 3).
#             edge_idx: Indices of center and neighbor atoms, that form the edges.
#                 Shape (2, n_edges). The first row is the center atom indices, and the
#                 second row is the neighbor atom indices.
#             atom_type: Atom types. Shape (Na,), where Na is the number of atoms.
#             atom_feats: shape (Na, F, T1), where Na is the number of atoms, F is the
#                 number of features, and T1 = (3**(L1+1)-1)//2 is the tensor dim.
#
#         Returns:
#             Updated atom feats. Shape (Na, F, T3), where T3 is the number tensor
#             components determined by L3.
#         """
#         assert (
#             atom_feats.shape[-1] == (3 ** (self.L1 + 1) - 1) // 2
#         ), "Invalid atom feats shape."
#
#         # Indices of center atoms (i) and neighbor atoms (j); (n_edges,)
#         i_idx = edge_idx[0]
#         j_idx = edge_idx[1]
#
#         # Polyadics of unit vectors; (n_edges, T2)
#         polyadics = get_polyadics_from_vector(
#             nn.functional.normalize(edge_vector, p=2.0, dim=-1), self.L2
#         )
#
#         # Radial basis; (n_edges, F)
#         fu = self.radial(
#             torch.linalg.vector_norm(edge_vector, dim=-1),
#             atom_type[i_idx],
#             atom_type[j_idx],
#         )
#
#         # TODO, the radial_mlp might be batched
#         #  put all weights as a larger one
#         # Radial params for each l2; each element in the list is of shape (n_edges, F)
#         R = [fn(fu) for fn in self.radial_mlp]
#
#         # Concatenate along tensor dim; (n_edges, F, T2)
#         R_cat = torch.cat(
#             [
#                 r.unsqueeze(-1).expand(-1, -1, 3**l2)
#                 for r, l2 in zip(R, range(self.L2 + 1))
#             ],
#             dim=-1,
#         )
#
#         # Evaluate RD or RhD; (n_edges, F, T2)
#         # R_cat: (n_edges, F, T2)
#         # polyadics: (n_edges, T2)
#         # atom_feats[j_idx][:1]: (n_edges, F, 1)
#         if self.mode == "variant1":
#             rp = R_cat * polyadics.unsqueeze(-2)
#         elif self.mode == "variant2":
#             rp = R_cat * atom_feats[j_idx][:1] * polyadics.unsqueeze(-2)
#         else:
#             raise ValueError(f"Invalid mode: {self.mode}")
#
#         # aggregate atoms j (neighbors) to atom i (center); (n_atoms, F, T2)
#         rp_sum = scatter(rp, i_idx, reduce="sum", dim=0) / self.num_average_neigh**0.5
#
#         # Tensor product of the features of atom i and the sum of rp of atoms j
#         # atom_feats[j_idx]: (atoms, F, T1)
#         # rp_sum: (n_atoms, F, T2)
#         # M: (n_atoms, F, T3)
#         M = self.tp(atom_feats, rp_sum)
#
#         return M
