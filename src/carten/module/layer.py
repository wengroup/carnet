"""CARTEN layer module."""

import torch
from torch import Tensor, nn

from .activation import elu, shifted_softplus, silu
from .atomic_moment import AtomicMoment, AtomicMoment2
from .hyper_moment import HyperMoment
from .linear import SlicedLinearMap
from .normalize import LayerNorm


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
        max_chebyshev_degree: int = 8,
        r_cut: float = 5.0,
        radial_mlp_hidden_layers: int | list[int] = 2,
        max_out_L: int = None,
        max_degree: int = None,
        atomic_moment_mode: str = "vanilla",
        layer_norm: bool = False,
        activation: str = None,
        residual: bool = True,
        use_linear_channel_input: bool = False,
        use_linear_channel_hyper: bool = False,
        use_linear_channel_residual: bool = True,
    ):
        """

        Args:
            F:
            L1:
            L2:
            L3:
            num_atom_types:
            num_average_neigh:
            max_chebyshev_degree:
            r_cut:
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
        self.max_chebyshev_degree = max_chebyshev_degree
        self.r_cut = r_cut
        self.radial_mlp_hidden_layers = radial_mlp_hidden_layers
        self.max_out_L = L3 if max_out_L is None else max_out_L
        self.max_degree = L3 if max_degree is None else max_degree
        self.residual = residual
        self.atomic_moment_mode = atomic_moment_mode

        # Kernel for mixing input atom feats across channel, separate for each rank
        if use_linear_channel_input:
            self.linear_channel_input = SlicedLinearMap(
                F, F, [3**l for l in range(L1 + 1)], bias=True
            )
        else:
            self.register_buffer("linear_channel_input", None)

        if atomic_moment_mode == "vanilla":
            AM = AtomicMoment
        elif atomic_moment_mode in ["variant1", "variant2"]:
            AM = AtomicMoment2
        else:
            raise ValueError(f"Unknown atomic_moment_mode: {atomic_moment_mode}")
        self.atomic_moment = AM(
            F=F,
            L1=L1,
            L2=L2,
            L3=L3,
            num_atom_types=num_atom_types,
            num_average_neigh=num_average_neigh,
            max_chebyshev_degree=max_chebyshev_degree,
            radial_mlp_hidden_layers=radial_mlp_hidden_layers,
            r_cut=r_cut,
            mode=atomic_moment_mode,
        )

        # Kernel for mixing channel of atomic moment, separate for each rank
        self.linear_channel_atomic = SlicedLinearMap(
            F, F, [3**l for l in range(L3 + 1)], bias=True
        )

        self.hyper_moment = HyperMoment(
            F=self.F, L=self.L3, max_out_L=self.max_out_L, max_degree=self.max_degree
        )

        # Kernel for mixing channel of hyper moment, separate for each rank
        if use_linear_channel_hyper:
            self.linear_channel_hyper = SlicedLinearMap(
                F, F, [3**l for l in range(self.max_out_L + 1)], bias=True
            )
        else:
            self.register_buffer("linear_channel_hyper", None)

        # Layer normalization
        if layer_norm:
            self.layer_norm = LayerNorm(
                dim=F,
                slice_sizes=[3**l for l in range(self.max_out_L + 1)],
            )
        else:
            self.register_buffer("layer_norm", None)

        # Nonlinear activation
        if activation is None:
            self.register_buffer("activation", None)
        elif activation == "elu":
            self.activation = silu
        elif activation == "silu":
            self.activation = elu
        elif activation == "shifted_softplus":
            self.activation = shifted_softplus
        else:
            supported = ["silu", "elu", "shifted_softplus"]
            raise ValueError(
                f"got unsupported activation function: {activation}. "
                f"Supported are: {supported}."
            )

        # If activation is used, add another linear layer after it
        if activation is not None:
            # If activation is used, the scalars in the above linear_channel_hyper and
            # layer_norm are used to create the gate scalar for the activation,
            # i.e. the scalars are used as x in g(x)*t.
            # Then, we create additional layer norm for the scalar features.
            # This is the same as the equiformer way of doing activation.
            if use_linear_channel_hyper:
                self.linear_channel_hyper_scalar = SlicedLinearMap(F, F, [1], bias=True)
            else:
                self.register_buffer("linear_channel_hyper_scalar", None)

            if layer_norm:
                self.layer_norm_scalar = LayerNorm(dim=F, slice_sizes=[1])
            else:
                self.register_buffer("layer_norm_scalar", None)

            self.linear_after_activation = SlicedLinearMap(
                F, F, [3**l for l in range(self.max_out_L + 1)], bias=True
            )
        else:
            self.register_buffer("linear_after_activation", None)

        # Residual connection, separate for each rank
        if self.residual:
            # Only do it for the ranks that exist in both the input atom feats and the
            # output hyper moment
            self.min_max_out_L_L1 = min(self.max_out_L, self.L1)

        if self.residual and use_linear_channel_residual:
            self.linear_channel_residual = SlicedLinearMap(
                F, F, [3**l for l in range(self.min_max_out_L_L1 + 1)], bias=True
            )
        else:
            self.register_buffer("linear_channel_residual", None)

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
            Updated atom feats. Shape (Na, F, T'), where T' is the number of tensor
            components, determined by max_out_L.
        """
        am = atom_feats  # (Na, F, T1)

        # Mixing input atom feats across channel
        if self.linear_channel_input is not None:
            am = self.linear_channel_input(am)  # (Na, F, T1)

        # Get atomic moments; (Na, F, T3)
        am = self.atomic_moment(edge_vector, edge_idx, atom_type, am)

        # Mix atomic moments across channel
        am = self.linear_channel_atomic(am)  # (Na, F, T3)

        # Get hyper moments; (Na, F, T')
        hm = self.hyper_moment(am)

        # Mix hyper moments across channel
        if self.linear_channel_hyper is not None:
            hm_2 = self.linear_channel_hyper(hm)  # (Na, F, T')
        else:
            hm_2 = hm

        # Normalize
        if self.layer_norm is not None:
            hm_2 = self.layer_norm(hm_2)

        # Activation
        if self.activation is None:
            hm = hm_2
        else:
            hm_2 = self.activation(hm_2)

            # If activation is not None, then self.linear_channel_hyper and
            # self.layer_norm are for the high-rank tensors. Generate it for scalars.
            if self.linear_channel_hyper_scalar is not None:
                hm_scalar = self.linear_channel_hyper_scalar(hm[..., 0:1])
            else:
                hm_scalar = hm[..., 0:1]

            if self.layer_norm_scalar is not None:
                hm_scalar = self.layer_norm_scalar(hm_scalar)

            # Apply activation to the scalar features
            hm_scalar = self.activation(hm_scalar)

            # hm_2 will have the scalar part due to the fact that we use the
            # general activation function g(x) * t, where x is the scalar features.
            # But, we want the scalar part to be separate, so we select the 1:
            # high-rank tensor part and concatenate with the separate scalar part.
            hm = torch.cat([hm_scalar, hm_2[..., 1:]], dim=-1)

            if self.linear_after_activation is not None:
                hm = self.linear_after_activation(hm)

        # Residual: mix input atom feats across channel and add to the output
        if self.residual:
            size = int((3 ** (self.min_max_out_L_L1 + 1) - 1) // 2)
            feats_skip = atom_feats[..., :size]

            if self.linear_channel_residual is not None:
                feats_skip = self.linear_channel_residual(feats_skip)

            hm[..., :size] += feats_skip

        return hm
