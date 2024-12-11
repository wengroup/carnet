import math

import torch
from torch import Tensor, nn


class LinearCombination(nn.Module):
    """
    Linear combination of tensors.

    Given a tensor of shape (..., d0, d1, d2), this module computes the linear
    combination along the dimension d0, but separately for each d1 dimension,
    resulting in a tensor of shape (..., d1, d2).

    Args:
        in_features: d0
        const_features: d1
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
            input: tensor of shape (...,d0, d1, d2)

        Returns:
            tensor of shape (..., d1, d2)
        """

        return torch.einsum("ij,...ijk->...jk", self.weight, input)


class LinearMap(nn.Module):
    """
    Linear map of tensors.

    Given a tensor of shape (..., d1, d2), this module computes the linear map of the
    tensor along the d1 (last but one) dimension, and returns a tensor of shape
    (..., d1', d2).

    Args:
        in_features: d1
        out_features: d1'
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
            input: tensor of shape (..., d1, d2)

        Returns:
            tensor of shape (..., d1', d2)
        """

        out = torch.einsum("ij,...jk->...ik", self.weight, input)
        if self.bias is not None:
            out += self.bias

        return out
