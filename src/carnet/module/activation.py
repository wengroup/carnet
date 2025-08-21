"""
Activation function for natural tensors.

This module is largely inspired by e3x activation:
https://e3x.readthedocs.io/stable/_autosummary/e3x.nn.activations.html#module-e3x.nn.activations
"""

import math
from typing import Callable

import torch
from torch import Tensor


def _gated_linear(g: Callable, x: Tensor) -> Tensor:
    """General gated linear activation.

    y = gate(x) * x

    Args:
         g: gate function
         x: input tensor. Shape (..., F, T), where F is the number of features,
            and T is the tensor dimension.

    Returns:
        Gated linear activation.
    """
    if len(x.shape) < 2:
        raise ValueError(
            "shape of x must have at least two dimensions, got " f"{x.shape}"
        )
    return g(x[..., :, 0:1]) * x


def relu(x: Tensor) -> Tensor:
    g = lambda x: torch.maximum(
        torch.sign(x), torch.tensor(0.0, device=x.device, dtype=x.dtype)
    )
    return _gated_linear(g, x)


def elu(x: Tensor, alpha: float = 1.0) -> Tensor:

    def g(x):
        not_tiny = torch.abs(x) > torch.finfo(x.dtype).eps
        expm1_safe_x = torch.where(
            x > 0, torch.tensor(0.0, dtype=x.dtype, device=x.device), x
        )
        div_safe_x = torch.where(
            not_tiny, x, torch.tensor(1.0, dtype=x.dtype, device=x.device)
        )

        return torch.where(
            x > 0,
            torch.tensor(1.0, dtype=x.dtype, device=x.device),
            torch.where(
                not_tiny,
                alpha * torch.expm1(expm1_safe_x) / div_safe_x,
                alpha * (x * x / 6 + x / 2 + 1),  # + O(x^3) Taylor series around x=0
            ),
        )

    return _gated_linear(g, x)


def silu(x: Tensor) -> Tensor:
    return _gated_linear(torch.sigmoid, x)


def shifted_softplus(x: Tensor) -> Tensor:
    def g(x):
        not_tiny = torch.abs(x) > torch.finfo(x.dtype).eps
        safe_x = torch.where(
            not_tiny, x, torch.tensor(1.0, dtype=x.dtype, device=x.device)
        )  # division safe x

        return torch.where(
            not_tiny,
            (
                torch.logaddexp(x, torch.tensor(0.0, dtype=x.dtype, device=x.device))
                - math.log(2.0)
            )
            / safe_x,
            x / 8.0 + 1.0 / 2.0,  # + O(x^3) Taylor series around x=0
        )

    return _gated_linear(g, x)
