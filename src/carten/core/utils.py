from typing import Optional

import torch
from torch import Tensor


def check_shape(T: Tensor, n: int = 3) -> bool:
    """Check a tensor is of shape (n, n, ..., n)"""
    if T.ndim == 0:
        return True
    elif set(T.shape) != {n}:
        return False
    else:
        return True


def factorial(n: int, device: Optional[torch.device] = None):
    """
    Get the factorial of a number.
    """
    return torch.prod(torch.arange(1, n + 1, device=device))


def double_factorial(
    n: int, lower_bound: Optional[int] = None, device: Optional[torch.device] = None
) -> Tensor:
    """
    Get the double factorial of a number.

    Args:
        n: The number to calculate the double factorial
        lower_bound: The lower bound of the double factorial. If lower bound is
            provided, this is calculated as n * (n-2) * ... * lower_bound. Default is
            None, meaning 1 if n odd and 2 if n even.
        device: The device to put the tensor on.
    """

    if n == 0 or n == 1:
        return torch.tensor(1, device=device)
    elif n % 2 == 0:
        if lower_bound is None:
            lower_bound = 2
        else:
            assert lower_bound % 2 == 0, "lower_bound must be even"
        return torch.prod(torch.arange(lower_bound, n + 2, step=2, device=device))
    else:
        if lower_bound is None:
            lower_bound = 1
        else:
            assert lower_bound % 2 == 1, "lower_bound must be odd"
        return torch.prod(torch.arange(lower_bound, n + 2, step=2, device=device))
