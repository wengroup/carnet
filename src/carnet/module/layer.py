"""CarNet layer module."""

from typing import Optional

from torch import Tensor, nn

from .atomic_moment import AtomicMoment
from .hyper_moment import HyperMoment
from .linear import SlicedLinearMap, SlicedLinearMap2


class Layer(nn.Module):
    """
    A layer consisting of an atomic moment module followed by a hyper moment module.
    """

    def __init__(
        self,
        F: int,
        L1: int,
        L2: int,
        L3: int,
        num_atom_types: int,
        num_average_neigh: float,
        radial_output_dim: int,
        radial_mlp_hidden_layers: int | list[int] = 2,
        max_out_L: int = None,
        max_degree: int = 3,
        tp_path_mode: str = "lite",
        tp_path_polar_only: bool = False,
        level: int = None,
        use_linear_channel_input: bool = False,
        use_linear_channel_hyper: bool = False,
        use_linear_channel_residual: bool = True,
        use_atomic_dependent_weight: bool | str = True,
        residual: bool = True,
        layer_index: int = None,
    ):
        """

        Args:
            F:
            L1:
            L2:
            L3:
            num_atom_types:
            num_average_neigh:
            radial_output_dim:
            radial_mlp_hidden_layers: if list of int, this gives the size of each hidden
                layer in the MLP that is applied to the radial basis functions. If int,
                this gives the number of hidden layers, and the size of each hidden
                layer is set to `F`, the channel dimension.
            max_out_L: Max rank for the output feature tensor. If None, set to L.
            max_degree: Max correlation degree of the hyper moment feature tensor.
                If None, set to L.
            activation: Nonlinear activation function to apply after each layer. If
                `None`, no activation is applied.
            residual: whether to use residual connection, that is, mixing the input
                atom feats across channel and adding to the output.
        """

        super().__init__()

        self.F = F
        self.L1 = L1
        self.L2 = L2
        self.L3 = L3
        self.num_atom_types = num_atom_types
        self.num_average_neigh = num_average_neigh
        self.radial_mlp_hidden_layers = radial_mlp_hidden_layers
        self.max_out_L = L3 if max_out_L is None else max_out_L
        self.max_degree = max_degree
        self.residual = residual
        self.use_atomic_dependent_weight = use_atomic_dependent_weight
        self.layer_index = layer_index

        # Process use atomic dependent weight option
        if isinstance(use_atomic_dependent_weight, bool):
            self.uadw_atomic_moment = use_atomic_dependent_weight
            self.uadw_residual = use_atomic_dependent_weight
        elif isinstance(use_atomic_dependent_weight, str):
            if use_atomic_dependent_weight.lower() == "atomic_moment":
                self.uadw_atomic_moment = True
                self.uadw_residual = False
            elif use_atomic_dependent_weight.lower() == "residual":
                self.uadw_atomic_moment = False
                self.uadw_residual = True
            else:
                expected = ["atomic_moment", "residual"]
                raise ValueError(
                    f"Expect use_atomic_dependent_weight to be a bool, or {expected}."
                    f"Got {use_atomic_dependent_weight}."
                )
        else:
            raise ValueError(
                "use_atomic_dependent_weight should be bool or str. "
                f"Got {type(use_atomic_dependent_weight)}."
            )

        # Kernel for mixing input atom feats across channel, separate for each rank
        if use_linear_channel_input:
            self.linear_channel_input = SlicedLinearMap(
                F, F, [3**l for l in range(L1 + 1)], bias=True
            )
        else:
            self.register_buffer("linear_channel_input", None)

        self.atomic_moment = AtomicMoment(
            F=F,
            L1=L1,
            L2=L2,
            L3=L3,
            num_atom_types=num_atom_types,
            num_average_neigh=num_average_neigh,
            radial_output_dim=radial_output_dim,
            radial_mlp_hidden_layers=radial_mlp_hidden_layers,
            tp_path_mode=tp_path_mode,
            tp_path_polar_only=tp_path_polar_only,
            level=level,
        )

        # Kernel for mixing channel of atomic moment, separate for each rank
        if self.uadw_atomic_moment:
            self.linear_channel_atomic = SlicedLinearMap2(
                F,
                F,
                [3**l for l in range(L3 + 1)],
                num_atom_types=num_atom_types,
                bias=True,
            )
        else:
            self.linear_channel_atomic = SlicedLinearMap(
                F, F, [3**l for l in range(L3 + 1)], bias=True
            )

        self.hyper_moment = HyperMoment(
            F=self.F,
            L=self.L3,
            max_out_L=self.max_out_L,
            max_degree=self.max_degree,
            tp_path_mode=tp_path_mode,
            tp_path_polar_only=tp_path_polar_only,
            level=level,
        )

        if use_linear_channel_hyper:
            self.linear_channel_hyper = SlicedLinearMap(
                F, F, [3**l for l in range(self.max_out_L + 1)], bias=True
            )
        else:
            self.register_buffer("linear_channel_hyper", None)

        # Residual connection, separate for each rank
        if self.residual:
            # Only do it for the ranks that exist in both the input atom feats and the
            # output hyper moment
            min_max_out_L_L1 = min(self.max_out_L, self.L1)
            self.residual_size = (3 ** (min_max_out_L_L1 + 1) - 1) // 2

        if self.residual and use_linear_channel_residual:
            if self.uadw_residual:
                self.linear_channel_residual = SlicedLinearMap2(
                    F,
                    F,
                    [3**l for l in range(min_max_out_L_L1 + 1)],
                    num_atom_types=num_atom_types,
                    bias=True,
                )
            else:
                self.linear_channel_residual = SlicedLinearMap(
                    F, F, [3**l for l in range(min_max_out_L_L1 + 1)], bias=True
                )
        else:
            self.register_buffer("linear_channel_residual", None)

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
            polyadics: Precomputed polyadic tensors of unit vectors. Shape (n_edges, T2).

        Returns:
            Updated atom feats. Shape (Na, F, T'), where T' is the number of tensor
            components, determined by max_out_L.
        """
        am = atom_feats  # (Na, F, T1)

        # Mixing input atom feats across channel
        if self.linear_channel_input is not None:
            am = self.linear_channel_input(am)  # (Na, F, T1)

        # Get atomic moments; (Na, F, T3)
        am = self.atomic_moment(
            edge_vector, edge_idx, atom_type, am, radial_basis, polyadics
        )

        # Mix atomic moments across channel
        if self.uadw_atomic_moment:
            am = self.linear_channel_atomic(am, atom_type)  # (Na, F, T3)
        else:
            am = self.linear_channel_atomic(am)  # (Na, F, T3)

        # Get hyper moments; (Na, F, T')
        hm = self.hyper_moment(am)

        # Mix hyper moments across channel
        if self.linear_channel_hyper is not None:
            hm = self.linear_channel_hyper(hm)  # (Na, F, T')

        # Residual: mix input atom feats across channel and add to the output
        if self.residual:
            feats_skip = atom_feats[..., : self.residual_size]

            if self.linear_channel_residual is not None:
                if self.uadw_residual:
                    feats_skip = self.linear_channel_residual(feats_skip, atom_type)
                else:
                    feats_skip = self.linear_channel_residual(feats_skip)

            hm[..., : self.residual_size] += feats_skip

        return hm
