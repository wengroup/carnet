"""
Lengendre polynomials.

See: https://en.wikipedia.org/wiki/Legendre_polynomials
"""

import torch
from torch import Tensor


def legendre(x: Tensor, n: int) -> Tensor:
    """
    Calculate the Legendre polynomial of degree n at x.

    Args:
        n: the degree of the Legendre polynomial
        x: the value at which to evaluate the polynomial

    Returns:
        The value of the Legendre polynomial at x.
    """
    if n == 0:
        return torch.tensor(1.0)
    if n == 1:
        return x

    return ((2 * n - 1) * x * legendre(x, n - 1) - (n - 1) * legendre(x, n - 2)) / n
