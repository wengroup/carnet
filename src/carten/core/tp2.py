"""Tensor product between two natural tensors.

Batching and feature dimensions of the tensors are supported.

This implements the formula such that Z = H:XY, where H is an operator composed of
delta and Levi-Civita tensors. Unlike `tp.py`, this reformulation does not require
loop to compute the tensor product.

Ref:
[LP89] "Angular reduction in multiparticle matrix elements" by D. R. Lehman and W. C. Parke.
http://dx.doi.org/10.1063/1.528515
"""

import torch
from line_profiler import profile
from natt.H_tp import get_H_numerical_even, get_H_numerical_odd
from torch import Tensor


@profile
def tp_even(
    X: Tensor, Y: Tensor, l1: int, l2: int, l3: int, normalize: str = "unity"
) -> Tensor:
    """
    Calculate the tensor product Z_l3 = X_l1 \otimes Y_l2 where l1 + l2 - l3 is even.

    Args:
        X: A natural tensor of rank l1. Shape: (..., F, 3^l1), where F is the number of
            features.
        Y: A natural tensor of rank l2. Shape: (..., F, 3^l2), where F is the number of
            features.
        l1: The rank of the first tensor X.
        l2: The rank of the second tensor Y.
        l3: The rank of the output tensor Z.
        normalize: The normalization method.
            If `unity`, the output is normalized such that the l3 fold contraction of
            the output tensor with a unit vector yields 1.
            If `none`, no normalization is applied.

    Returns:
        A natural tensor of rank l3. Shape: (..., F, 3^l3), where F is the number of
        features.
    """

    assert (l1 + l2 - l3) % 2 == 0, "l1 + l2 - l3 must be even"

    leading_dims = X.shape[:-1]  # including the feature dimension

    # TODO, correct the dtype and device of H
    dtype = X.dtype
    device = X.device

    # Expand the tensors dims: (..., 3^l1) -> (..., 3, 3, ..., 3)
    X = X.view(leading_dims + (3,) * l1)
    Y = Y.view(leading_dims + (3,) * l2)

    # Performing tensor product
    H, rule = get_H_numerical_even(l1, l2, l3, normalize)
    Z = torch.einsum(rule, H, X, Y)

    # Combine the tensor dims: (..., 3, 3, ..., 3) -> (..., 3^l3)
    Z = Z.view(leading_dims + (3**l3,))

    return Z


@profile
def tp_odd(
    X: Tensor, Y: Tensor, l1: int, l2: int, l3: int, normalize: str = "unity"
) -> Tensor:
    """
    Calculate the tensor product Z_l3 = X_l1 \otimes Y_l2 where l1 + l2 - l3 is odd.

    Args:
        X: A natural tensor of rank l1. Shape: (..., F, 3^l1), where F is the number of
            features.
        Y: A natural tensor of rank l2. Shape: (..., F, 3^l2), where F is the number of
            features.
        l1: The rank of the first tensor X.
        l2: The rank of the second tensor Y.
        l3: The rank of the output tensor Z.

    Returns:
        A natural tensor of rank l3. Shape: (..., F, 3^l3), where F is the number of
        features.
    """
    assert (l1 + l2 - l3) % 2 == 1, "l1 + l2 - l3 must be odd"

    leading_dims = X.shape[:-1]  # including the feature dimension

    # TODO, correct the dtype and device of H
    dtype = X.dtype
    device = X.device

    # Expand the tensors dims: (..., 3^l1) -> (..., 3, 3, ..., 3)
    X = X.view(leading_dims + (3,) * l1)
    Y = Y.view(leading_dims + (3,) * l2)

    # Performing tensor product
    H, rule = get_H_numerical_odd(l1, l2, l3, normalize)
    Z = torch.einsum(rule, H, X, Y)

    # Combine the tensor dims: (..., 3, 3, ..., 3) -> (..., 3^l3)
    Z = Z.view(leading_dims + (3**l3,))

    return Z


if __name__ == "__main__":
    from natt.symmetrize import get_random_natural_tensor

    l1 = 4
    l2 = 4
    l3 = 4
    l4 = 3

    X = get_random_natural_tensor(l1, seed=1).view(-1)
    Y = get_random_natural_tensor(l2, seed=2).view(-1)

    tp_even(X, Y, l1, l2, l3)
    tp_odd(X, Y, l1, l2, l4)
