"""Tensor product between two natural tensors.

Batching and feature dimensions of the tensors are supported.

This is the reference implementation of the tensor product, which can used to verify the
performance of the optimized implementations in `tp2.py`, `tp3.py` and `tp4.py`.

Ref:
[LP89] "Angular reduction in multiparticle matrix elements" by D. R. Lehman and W. C. Parke.
http://dx.doi.org/10.1063/1.528515
"""

import torch
from line_profiler import profile
from natt.H_tp import coeff_C, coeff_D, get_tp_even_rule, get_tp_odd_rule
from natt.symmetrize import get_permutations_delta, symmetrize_via_permutation
from natt.utils import dij, double_factorial, eijk
from torch import Tensor

# CACHE to speed up calculation, they will be filled when the functions are called
TP_EVEN_RULE_CACHE = {}
TP_ODD_RULE_CACHE = {}
PERMUTATIONS_DELTA_CACHE = {}
COEFF_C_CACHE = {}
COEFF_D_CACHE = {}


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
    assert abs(l1 - l2) <= l3 <= l1 + l2, "l3 must be in the range of |l1-l2| and l1+l2"
    assert (l1 + l2 - l3) % 2 == 0, "l1 + l2 - l3 must be even"

    leading_dims = X.shape[:-1]  # including the feature dimension
    dtype = X.dtype
    device = X.device

    k = (l1 + l2 - l3) // 2

    d = dij(device)

    # Expand the tensors dims: (..., 3^l1) -> (..., 3, 3, ..., 3)
    X = X.view(leading_dims + (3,) * l1)
    Y = Y.view(leading_dims + (3,) * l2)

    # Create the output tensor
    Z = torch.zeros(leading_dims + (3,) * l3, dtype=dtype, device=device)

    for t in range(min(l1, l2) - k + 1):
        coeff = (-2) ** t / double_factorial(
            2 * l3 - 1, 2 * l3 - 2 * t - 1 + 2, device=device
        )

        global TP_EVEN_RULE_CACHE
        if (l1, l2, k, t) not in TP_EVEN_RULE_CACHE:
            rule, symmetry, delta_indices = get_tp_even_rule(l1, l2, k, t)
            TP_EVEN_RULE_CACHE[(l1, l2, k, t)] = (rule, symmetry, delta_indices)
        else:
            rule, symmetry, delta_indices = TP_EVEN_RULE_CACHE[(l1, l2, k, t)]

        # Get one tensor product
        operands = [X, Y] + [d] * t
        prod = torch.einsum(rule, operands)

        # Symmetrize by summing over all unique permutations
        global PERMUTATIONS_DELTA_CACHE
        nd = len(leading_dims)
        if (symmetry, delta_indices, nd) not in PERMUTATIONS_DELTA_CACHE:
            perms = get_permutations_delta(symmetry, delta_indices, nd)
            PERMUTATIONS_DELTA_CACHE[(symmetry, delta_indices, nd)] = perms
        else:
            perms = PERMUTATIONS_DELTA_CACHE[(symmetry, delta_indices, nd)]

        prod = symmetrize_via_permutation(prod, perms, mode="sum")

        Z += coeff * prod

    if normalize == "unity":
        global COEFF_C_CACHE
        if (l1, l2, l3) not in COEFF_C_CACHE:
            c = coeff_C(l1, l2, l3, device=device)
            COEFF_C_CACHE[(l1, l2, l3)] = c
        else:
            c = COEFF_C_CACHE[(l1, l2, l3)]
        Z *= c
    elif normalize == "none":
        pass
    else:
        supported = ["unity", "none"]
        raise ValueError(
            f"Unknown normalization method: {normalize}. Supported are: {supported}"
        )

    # Combine the tensor dims: (..., 3, 3, ..., 3) -> (..., 3^l3)
    Z = Z.view(leading_dims + (-1,))

    return Z


@profile
def tp_odd(
    X: Tensor, Y: Tensor, l1: int, l2: int, l3: int, normalize: str = "unity"
) -> Tensor:
    """
    Calculate the tensor product of two natural tensors, when l1 + l2 - l3 is odd.

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
    assert abs(l1 - l2) <= l3 <= l1 + l2, "l3 must be in the range of |l1-l2| and l1+l2"
    assert (l1 + l2 - l3) % 2 == 1, "l1 + l2 - l3 must be odd"

    leading_dims = X.shape[:-1]
    dtype = X.dtype
    device = X.device

    k = (l1 + l2 - l3 - 1) // 2

    d = dij(device)
    epsilon = eijk(device)

    # Expand the tensors dims: (..., 3^l1) -> (..., 3, 3, ..., 3)
    X = X.view(leading_dims + (3,) * l1)
    Y = Y.view(leading_dims + (3,) * l2)

    # Create the output tensor
    Z = torch.zeros(leading_dims + (3,) * l3, dtype=dtype, device=device)

    for t in range(min(l1, l2) - k):
        coeff = (-2) ** t / double_factorial(
            2 * l3 - 1, 2 * l3 - 2 * t - 1 + 2, device=device
        )

        global TP_ODD_RULE_CACHE
        if (l1, l2, k, t) not in TP_ODD_RULE_CACHE:
            # TODO, can move the calling of get_permutations_delta into
            #  get_tp_odd_rule and such
            rule, symmetry, delta_indices = get_tp_odd_rule(l1, l2, k, t)
            TP_ODD_RULE_CACHE[(l1, l2, k, t)] = (rule, symmetry, delta_indices)
        else:
            rule, symmetry, delta_indices = TP_ODD_RULE_CACHE[(l1, l2, k, t)]

        # Get one tensor product
        operands = [epsilon, X, Y] + [d] * t
        prod = torch.einsum(rule, operands)

        # Symmetrize by summing over all unique permutations
        global PERMUTATIONS_DELTA_CACHE
        nd = len(leading_dims)
        if (symmetry, delta_indices, nd) not in PERMUTATIONS_DELTA_CACHE:
            perms = get_permutations_delta(symmetry, delta_indices, nd)
            PERMUTATIONS_DELTA_CACHE[(symmetry, delta_indices, nd)] = perms
        else:
            perms = PERMUTATIONS_DELTA_CACHE[(symmetry, delta_indices, nd)]

        prod = symmetrize_via_permutation(prod, perms, mode="sum")

        Z += coeff * prod

    if normalize == "unity":
        global COEFF_D_CACHE
        if (l1, l2, l3) not in COEFF_D_CACHE:
            c = coeff_D(l1, l2, l3, device=device)
            COEFF_D_CACHE[(l1, l2, l3)] = c
        else:
            c = COEFF_D_CACHE[(l1, l2, l3)]
        Z *= c
    elif normalize == "none":
        pass
    else:
        raise ValueError(f"Unknown normalization method: {normalize}")

    # Combine the tensor dims: (..., 3, 3, ..., 3) -> (..., 3^l3)
    Z = Z.view(leading_dims + (-1,))

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
