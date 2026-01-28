import math

import torch
from torch import Tensor, nn


class LinearCombination(nn.Module):
    """
    Linear combination of tensors.

    Given a tensor of shape (..., P, F, t), this module computes the linear combination
    along the dimension P, but separately for each F dimension, resulting in a tensor
    of shape (..., F, t).

    Args:
        in_features: P
        const_features: F
    """

    def __init__(self, in_features: int, const_features: int):
        super().__init__()
        self.in_features = in_features
        self.const_features = const_features

        self.weight = nn.Parameter(torch.empty(in_features, const_features))
        self.reset_parameters()

    def reset_parameters(self):
        """
        https://github.com/pytorch/pytorch/blob/e3ca7346ce37d756903c06e69850bdff135b6009/torch/nn/modules/linear.py#L109
        """
        # We don't directly use nn.init.kaiming_uniform_ since in self.weight,
        # the in_features is not the last dimension
        k = 1 / self.in_features**0.5
        nn.init.uniform_(self.weight, -k, k)

    def forward(self, input: Tensor) -> Tensor:
        """
        Args:
            input: tensor of shape (...,P, F, t)

        Returns:
            tensor of shape (..., F, t)
        """

        return torch.einsum("pf,...pft->...ft", self.weight, input)

    def __repr__(self):
        return f"LinearCombination(in_features={self.in_features}, const_features={self.const_features})"


class SlicedLinearCombination(nn.Module):
    """
    Sliced linear combination of tensors.

    Given a tensor of shape (..., P, F, t), this module computes the linear combination
    along the dimension P, resulting in a tensor of shape (..., F, t).

    Unlike LinearCombination, where a single weight matrix is used for the entire tensor,
    here separate weight matrices are used for each slice across the t dimension.
    """

    def __init__(self, in_features: int, const_features: int, slice_sizes: list[int]):
        super().__init__()
        self.in_features = in_features
        self.const_features = const_features
        self.slice_sizes = slice_sizes

        self.num_slices = len(slice_sizes)
        # Separate weights for each slice
        self.weight = nn.Parameter(
            torch.empty(self.num_slices, in_features, const_features)
        )

        # Mapping from t index to slice index
        weight_map = torch.repeat_interleave(torch.tensor(slice_sizes))
        self.register_buffer("weight_map", weight_map)

        self.reset_parameters()

    def reset_parameters(self):
        k = 1 / self.in_features**0.5
        nn.init.uniform_(self.weight, -k, k)

    def forward(self, input: Tensor) -> Tensor:
        """
        Args:
            input: tensor of shape (..., P, F, t)

        Returns:
            tensor of shape (..., F, t)
        """
        # Expand weights to (t, P, F)
        W_full = self.weight[self.weight_map]

        # (t, p, f) * (..., p, f, t) -> (..., f, t)
        # We use einsum to handle the shared t dimension correctly
        return torch.einsum("tpf, ...pft -> ...ft", W_full, input)

    def __repr__(self):
        return (
            f"SlicedLinearCombination(in_features={self.in_features}, "
            f"const_features={self.const_features}, "
            f"slice_sizes={self.slice_sizes})"
        )


class LinearMap(nn.Module):
    """
    Linear map of tensors.

    Given a tensor of shape (..., F, t), this module computes the linear map of the
    tensor along the F (last but one) dimension, and returns a tensor of shape
    (..., F', t).

    Args:
        in_features: F
        out_features: F'
        bias: whether to add bias
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(torch.empty(out_features, in_features))

        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        """
        The same as Linear in pytorch.
        https://github.com/pytorch/pytorch/blob/e3ca7346ce37d756903c06e69850bdff135b6009/torch/nn/modules/linear.py#l109
        """
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input: Tensor) -> Tensor:
        """
        Args:
            input: tensor of shape (..., F, t)

        Returns:
            tensor of shape (..., F', t)
        """

        # out = torch.einsum("ij,...jt->...it", self.weight, input)
        out = torch.matmul(self.weight, input)

        if self.bias is not None:
            out += self.bias.unsqueeze(-1)

        return out

    def __repr__(self):
        return (
            f"LinearMap(in_features={self.in_features}, "
            f"out_features={self.out_features}, bias={self.bias is not None})"
        )


class LinearMap2(nn.Module):
    """
    Linear map of tensors.

    Similar to LinearMap, but the weight matrix is different for atoms of different
    species.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_atom_types: int,
        bias: bool = False,
    ):

        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_atom_types = num_atom_types

        self.weight = nn.Parameter(
            torch.empty(num_atom_types, out_features, in_features)
        )

        if bias:
            self.bias = nn.Parameter(torch.empty(num_atom_types, out_features))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        # init each of one of num_atom_types (dim 0) is the same as init them together
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        if self.bias is not None:
            bound = 1 / math.sqrt(self.in_features)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input: Tensor, atom_type: Tensor) -> Tensor:
        """
        Args:
            input: shape (N, F, t), where F is the in_features.
            atom_type: shape (N,)

        Returns:
            shape (N, F', t), where F' is the out_features.
        """
        # self.weight: (num_atom_types, out_features, in_features)
        # W: (N, out_features, in_features)
        W = self.weight[atom_type]

        # out = torch.einsum("Noi,Nit->Not", W, input)  # (N, out_features, t)
        out = torch.bmm(W, input)  # (N, out_features, t)

        if self.bias is not None:
            b = self.bias[atom_type]  # (N, out_features)
            out += b.unsqueeze(-1)

        return out


class SlicedLinearMap(nn.Module):
    """
    Sliced linear map of tensors.

    Given a tensor of shape (..., F, T), this module computes the linear map of the
    tensor along the F (last but one) dimension to a tensor of shape (..., F', T).

    Unlike LinearMap, where a single weight matrix is used for the entire tensor, here
    separate weight matrices are used for each slice across the T dimension.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        slice_sizes: list[int],
        bias: bool = True,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.slice_sizes = slice_sizes
        self.bias = bias

        self.num_slices = len(slice_sizes)
        # Separate weights for each slice
        self.weight = nn.Parameter(
            torch.empty(self.num_slices, out_features, in_features)
        )

        # Bias only for the first slice, which should be a scalar (size 1)
        if bias:
            if slice_sizes[0] != 1 or any(s == 1 for s in slice_sizes[1:]):
                raise ValueError(
                    f"Bias can only be added if the first slice is a scalar (size 1) "
                    f"and it is the only scalar. Got slice_sizes {slice_sizes}."
                )
            self.bias_param = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias_param", None)

        # Mapping from T index to slice index
        weight_map = torch.repeat_interleave(torch.tensor(slice_sizes))
        self.register_buffer("weight_map", weight_map)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias_param is not None:
            fan_in = self.in_features
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias_param, -bound, bound)

    def forward(self, input: Tensor) -> Tensor:
        """
        Args:
            input: tensor of shape (..., F, T)

        Returns:
            tensor of shape (..., F', T)
        """
        # Expand weights to (T, out_F, in_F)
        W_full = self.weight[self.weight_map]

        # (T, out_F, in_F) * (..., in_F, T) -> (..., out_F, T)
        out = torch.einsum("tif, ...ft -> ...it", W_full, input)

        if self.bias_param is not None:
            # Add bias only to the first entry of the T dimension (the scalar)
            out[..., 0] = out[..., 0] + self.bias_param

        return out

    def __repr__(self):
        return (
            f"SlicedLinearMap(in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"slice_sizes={self.slice_sizes}, "
            f"bias={self.bias is not None})"
        )


class SlicedLinearMap2(nn.Module):
    """
    Sliced linear map of tensors.

    Similar to SlicedLinearMap, but the weight matrix is different for atoms of
    different species.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        slice_sizes: list[int],
        num_atom_types: int,
        bias: bool = True,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.slice_sizes = slice_sizes
        self.bias = bias
        self.num_atom_types = num_atom_types

        self.slices = []
        self.linear = nn.ModuleList()
        start = 0
        for size in slice_sizes:
            end = start + size
            self.slices.append(slice(start, end))
            start = end

            # Apply bias for scalars
            if bias and size == 1:
                b = True
            else:
                b = False

            self.linear.append(
                LinearMap2(
                    in_features=in_features,
                    out_features=out_features,
                    num_atom_types=num_atom_types,
                    bias=b,
                )
            )

    def forward(self, input: Tensor, atom_type: Tensor) -> Tensor:
        """
        Args:
            input: tensor of shape (..., F, T)
            atom_type: shape (N,)

        Returns:
            tensor of shape (..., F', T)
        """

        out = []
        for layer, s in zip(self.linear, self.slices):
            out.append(layer(input[..., s], atom_type))
        out = torch.cat(out, dim=-1)  # Shape (..., F', T)
        return out
