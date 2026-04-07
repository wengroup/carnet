"""Atomic moment module."""

from typing import Optional

import torch
from torch import Tensor, nn

from carnet.core.unit_vector import Polyadics

from .mlp import MLP
from .product import TensorProduct
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
        radial_output_dim: int,
        radial_mlp_hidden_layers: list[int] | int = 2,
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

        # `persistent=False` to not save in state dict, so that they can be overridden
        # when loading the model, e.g., for finetuning.
        self.register_buffer(
            "inv_sqrt_num_average_neigh",
            torch.as_tensor(1.0 / num_average_neigh**0.5),
            persistent=False,
        )

    def forward(
        self,
        edge_vector: Tensor,
        edge_idx: Tensor,
        atom_type: Tensor,
        atom_feats: Tensor,
        radial_basis: Tensor,
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
            radial_basis: Precomputed shared radial basis. Shape (n_edges, radial_output_dim).
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

        # Radial params for all paths; (n_edges, F * n_paths)
        R_all = self.radial_mlp(radial_basis)

        # Reshape to (n_edges, n_paths, F) to match TensorProduct interface.
        R_all = R_all.view(-1, len(self.all_paths), self.F)

        # Tensor product of the features of center atoms and neighbor atoms
        # atom_feats[j_idx]: (n_edges, F, T1)
        # polyadics: (n_edges, T2)
        # R_all: (n_edges, n_paths, F)
        # product: (n_edges, F, T3)
        product = self.tp(atom_feats[j_idx], polyadics, R_all)

        # aggregate atoms j (src) to atom i (dst); (n_atoms, F, T3)
        M = (
            scatter(product, i_idx, reduce="sum", dim=0, dim_size=atom_feats.shape[0])
            * self.inv_sqrt_num_average_neigh
        )

        return M
