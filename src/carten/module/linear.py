import math

import torch
from torch import Tensor, nn


class LinearCombination(nn.Module):
    """
    Linear combination of tensors.

    Given a tensor of shape (..., P, F, t), this module computes the linear combination
    along the dimension F', but separately for each F dimension, resulting in a tensor
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

        # Weight matrices, each for a slice; (F', F, 1)
        self.weights = nn.ParameterList(
            [
                nn.Parameter(torch.empty(out_features, in_features, 1))
                for _ in slice_sizes
            ]
        )

        # Bias for the first slice
        if bias:
            if slice_sizes[0] != 1:
                raise ValueError(
                    f"Bias can only be used when first slice size is 1 "
                    f"(namely a scalar), but got {slice_sizes[0]}"
                )
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        """
        https://github.com/pytorch/pytorch/blob/e3ca7346ce37d756903c06e69850bdff135b6009/torch/nn/modules/linear.py#l109
        """
        for w in self.weights:
            nn.init.kaiming_uniform_(w, a=math.sqrt(5))

        if self.bias is not None:
            fan_in = self.in_features
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input: Tensor) -> Tensor:
        """
        Args:
            input: tensor of shape (..., F, T)

        Returns:
            tensor of shape (..., F', T)
        """

        # Combine all weights into a single tensor; (F', F, T)
        expanded_weights = []
        for i, w in enumerate(self.weights):
            expanded_weights.append(w.expand(-1, -1, self.slice_sizes[i]))
        expanded_weight = torch.cat(expanded_weights, dim=-1)

        out = torch.einsum("ijt,...jt->...it", expanded_weight, input)

        if self.bias is not None:
            # Bias is now shape (F',) and we add it to first slice only
            out[..., 0] += self.bias

        return out
