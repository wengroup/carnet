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

        out = torch.einsum("ij,...jt->...it", self.weight, input)
        if self.bias is not None:
            out += self.bias.unsqueeze(-1)

        return out

    def __repr__(self):
        return (
            f"LinearMap(in_features={self.in_features}, "
            f"out_features={self.out_features}, bias={self.bias is not None})"
        )


class SlicedLinearMap(nn.Module):
    """
    Sliced linear map of tensors.

    Given a tensor of shape (..., F, T), this module computes the linear map of the
    tensor along the F (last but one) dimension to a tensor of shape (..., F', T).

    Unlike LinearMap, where a single weight matrix is used for the entire tensor, here
    separate weight matrices are used for each slice across the T dimension.

    For example, let T = t1+t2+...+tn, then n weight matrices are used, each for a slice
    of size ti across the T dimension.

    This is essentially a generalization of LinearMap.

    Args:
        in_features: F
        out_features: F'
        slice_sizes: Sizes of slices across the T dimension.
        bias: Whether to add bias for the first slice. Bias is only added to the first
            slice and when slice_sizes[0] == 1, namely when the first slice is a scalar.
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
                LinearMap(
                    in_features=in_features,
                    out_features=out_features,
                    bias=b,
                )
            )

    def forward(self, input: Tensor) -> Tensor:
        """
        Args:
            input: tensor of shape (..., F, T)

        Returns:
            tensor of shape (..., F', T)
        """

        out = []
        for layer, s in zip(self.linear, self.slices):
            out.append(layer(input[..., s]))
        out = torch.cat(out, dim=-1)  # Shape (..., F', T)
        return out

    def __repr__(self):
        return (
            f"SlicedLinearMap(in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"slice_sizes={self.slice_sizes}, "
            f"bias={self.bias is not None})"
        )
