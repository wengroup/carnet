"""Tensor product between two natural tensors.

This uses the explicit formula given in
"Angular reduction in multiparticle matrix elements" by D. R. Lehman and W. C. Parke.
http://dx.doi.org/10.1063/1.528515
"""

import torch
from torch import Tensor

from carten.reduce import symmetrize
from carten.utils import (
    dij,
    double_factorial,
    double_index,
    eijk,
    factorial,
    letter_index,
    repeat_double_index,
)


def tp_even(S: Tensor, T: Tensor, out_rank: int) -> Tensor:
    """
    Calculate the tensor product of two natural tensors, when l1 + l2 - l3 is even.

    Args:
        S: A natural tensor of rank l1
        T: A natural tensor of rank l2
        out_rank: The rank of the output tensor l3

    Returns:
        A natural tensor of rank l3
    """
    l1 = S.ndim
    l2 = T.ndim
    l3 = out_rank
    dtype = S.dtype
    device = S.device

    assert (l1 + l2 - l3) % 2 == 0, "l1 + l2 - l3 must be even"

    k = (l1 + l2 - l3) // 2
    out = torch.zeros([3] * l3, dtype=dtype, device=device)

    d = dij(device)

    for m in range(min(l1, l2) - k + 1):
        coeff = (-2) ** m / double_factorial(
            2 * l3 - 1, 2 * l3 - 2 * m - 1 + 2, device=device
        )

        rule, symmetry = tp_rule_even(l1, l2, k, m)
        prod = torch.einsum(rule, S, T, *([d] * m))
        prod = symmetrize(prod, symmetry=symmetry)

        out = out + coeff * prod

    out = coeff_C(l1, l2, l3) * out

    return out


def tp_odd(S: Tensor, T: Tensor, out_rank: int) -> Tensor:
    """
    Calculate the tensor product of two natural tensors, when l1 + l2 - l3 is odd.

    Args:
        S: A natural tensor of rank l1
        T: A natural tensor of rank l2
        out_rank: The rank of the output tensor l3

    Returns:
        A natural tensor of rank l3
    """
    l1 = S.ndim
    l2 = T.ndim
    l3 = out_rank
    dtype = S.dtype
    device = S.device

    assert (l1 + l2 - l3) % 2 == 1, "l1 + l2 - l3 must be odd"

    k = (l1 + l2 - l3 - 1) // 2
    out = torch.zeros([3] * l3, dtype=dtype, device=device)

    d = dij(device)
    epsilon = eijk(device)

    for m in range(min(l1, l2) - k):
        coeff = (-2) ** m / double_factorial(
            2 * l3 - 1, 2 * l3 - 2 * m - 1 + 2, device=device
        )

        rule, symmetry = tp_rule_odd(l1, l2, k, m)
        prod = torch.einsum(rule, epsilon, S, T, *([d] * m))
        prod = symmetrize(prod, symmetry=symmetry)

        out = out + coeff * prod

    out = coeff_D(l1, l2, l3) * out

    return out


def coeff_C(l1: int, l2: int, l3: int, device: torch.device = None):
    """Coefficient C for even L."""
    L = l1 + l2 + l3
    L1 = L - 2 * l1 - 1
    L2 = L - 2 * l2 - 1
    L3 = L - 2 * l3 - 1

    return (
        factorial(l1, device)
        * factorial(l2, device)
        * double_factorial(2 * l3 - 1, device=device)
        * factorial((L1 + 1) // 2, device=device)
        * factorial((L2 + 1) // 2, device=device)
        / factorial(l3, device=device)
        / double_factorial(L1, device=device)
        / double_factorial(L2, device=device)
        / double_factorial(L3, device=device)
        / factorial(L // 2, device=device)
    )


def coeff_D(l1: int, l2: int, l3: int, device: torch.device = None):
    """Coefficient D for odd L."""
    L = l1 + l2 + l3
    L1 = L - 2 * l1 - 1
    L2 = L - 2 * l2 - 1
    L3 = L - 2 * l3 - 1

    return (
        2
        * factorial(l1, device)
        * factorial(l2, device)
        * double_factorial(2 * l3 - 1, device=device)
        * factorial(L1 // 2, device=device)
        * factorial(L2 // 2, device=device)
        / factorial(l3 - 1, device=device)
        / double_factorial(L1 + 1, device=device)
        / double_factorial(L2 + 1, device=device)
        / double_factorial(L3 + 1, device=device)
        / factorial((L + 1) // 2, device=device)
    )


def tp_rule_even(l1: int, l2: int, k: int, m: int) -> tuple[str, str]:
    """
    Get the einsum rule when l1 + l2 - l3 is even.

    x_l1 \otimes^{k+m} x_l2 \otimes I ^{\otimes^m}

    After contraction, the resultant tensor will have l1-k-m indices from x, and these
    indices are still symmetric. Similarly, the resultant tensor will have l2-k-m
    symmetric indices from y. It will have 2*m indices from I. Each two indices from I
    are symmetric.

    In total, the resultant tensor will have l3 = l1 + l2 - 2k - 2m indices.

    Returns:
        rule: The einsum rule for the tensor product
        symmetry: The symmetry information of the resultant tensor after the tensor
            product. e.g. `aabb` means the first two indices are symmetric, and the last
            two indices are symmetric.
    """

    xy_contracted = letter_index(k + m)
    x_remain = letter_index(l1 - k - m, k + m)
    y_remain = letter_index(l2 - k - m, l1)

    # indices for contracting m I
    # x and y uses l1 + l2 - k - m indices
    delta = double_index(m, l1 + l2 - k - m)
    delta_left = "," + ",".join(delta) if delta else ""
    delta_right = "".join(delta)

    rule = (
        f"{xy_contracted}{x_remain},{xy_contracted}{y_remain}{delta_left}"
        f"->{x_remain}{y_remain}{delta_right}"
    )

    # l1-k-m remaining symmetric indices from x
    # l2-k-m remaining symmetric indices from y
    # 2m indices from all deltas. Each delta has 2 symmetric indices.
    symmetry = (
        "x" * len(x_remain) + "y" * len(y_remain) + "".join(repeat_double_index(m))
    )

    return rule, symmetry


def tp_rule_odd(l1: int, l2: int, k: int, m: int) -> tuple[str, str]:
    """
    Get the einsum rule when l1 + l2 - l3 is odd.

    epsilon : x_l1 \otimes^{k+m} x_l2 \otimes I ^{\otimes^m}

    epsilon is the Levi-Civita symbol. It contracts away one index from x and one index
    from y. So, after contraction, the resultant tensor will have 1 index from epsilon.
    After contraction, the resultant tensor will have l1-1-k-m indices from x, and these
    indices are still symmetric. Similarly, the resultant tensor will have l2-1-k-m
    symmetric indices from y. It will have 2*m indices from I. Each two indices from I
    are symmetric.

    In total, the resultant tensor will have l3 = l1 + l2 - 1 - 2k - 2m indices.


    Returns:
        rule: The einsum rule for the tensor product
        symmetry: The symmetry information of the resultant tensor after the tensor
            product. e.g. `aabb` means the first two indices are symmetric, and the last
            two indices are symmetric.
    """
    # example: epsilon_abc x_bpqr  y_cpqs I_tu -> arstu

    # epsilon_remain = "a"
    # epsilon_contracted = "bc"
    xy_contracted = letter_index(k + m, 3)
    x_remain = letter_index(l1 - 1 - k - m, k + m + 3)
    y_remain = letter_index(l2 - 1 - k - m, l1 + 2)

    # indices for contracting m I
    # x and y uses l1 + l2 - k - m indices
    delta = double_index(m, l1 + l2 - k - m + 1)
    delta_left = "," + ",".join(delta) if delta else ""
    delta_right = "".join(delta)

    rule = (
        f"abc,b{xy_contracted}{x_remain},c{xy_contracted}{y_remain}{delta_left}"
        f"->a{x_remain}{y_remain}{delta_right}"
    )

    # 1 index from epsilon
    # l1-1-k-m remaining symmetric indices from x
    # l2-1-k-m remaining symmetric indices from y
    # 2m indices from all deltas. Each delta has 2 symmetric indices.
    symmetry = (
        "x"
        + "y" * len(x_remain)
        + "z" * len(y_remain)
        + "".join(repeat_double_index(m))
    )

    return rule, symmetry
