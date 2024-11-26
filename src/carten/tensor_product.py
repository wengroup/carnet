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


def tp_even(X: Tensor, Y: Tensor, out_rank: int) -> Tensor:
    """
    Calculate the tensor product of two natural tensors, when l1 + l2 - l3 is even.

    Args:
        X: A natural tensor of rank l1
        Y: A natural tensor of rank l2
        out_rank: The rank of the output tensor l3

    Returns:
        A natural tensor of rank l3
    """
    l1 = X.ndim
    l2 = Y.ndim
    l3 = out_rank
    dtype = X.dtype
    device = X.device

    assert (l1 + l2 - l3) % 2 == 0, "l1 + l2 - l3 must be even"

    k = (l1 + l2 - l3) // 2
    d = dij(device)

    out = torch.zeros([3] * l3, dtype=dtype, device=device)

    for t in range(min(l1, l2) - k + 1):
        # TODO, since in symmetrize() we already considers average over all possible
        #  permutations, it seems the **m is not needed here.
        coeff = (-2) ** t / double_factorial(
            2 * l3 - 1, 2 * l3 - 2 * t - 1 + 2, device=device
        )

        rule, symmetry = tp_rule_even(l1, l2, k, t)
        prod = torch.einsum(rule, X, Y, *([d] * t))
        prod = symmetrize(prod, symmetry=symmetry)

        out = out + coeff * prod

    out = coeff_C(l1, l2, l3) * out

    return out


def tp_odd(X: Tensor, Y: Tensor, out_rank: int) -> Tensor:
    """
    Calculate the tensor product of two natural tensors, when l1 + l2 - l3 is odd.

    Args:
        X: A natural tensor of rank l1
        Y: A natural tensor of rank l2
        out_rank: The rank of the output tensor l3

    Returns:
        A natural tensor of rank l3
    """
    l1 = X.ndim
    l2 = Y.ndim
    l3 = out_rank
    dtype = X.dtype
    device = X.device

    assert (l1 + l2 - l3) % 2 == 1, "l1 + l2 - l3 must be odd"

    k = (l1 + l2 - l3 - 1) // 2
    out = torch.zeros([3] * l3, dtype=dtype, device=device)

    d = dij(device)
    epsilon = eijk(device)

    for t in range(min(l1, l2) - k):
        # TODO, since in symmetrize() we already considers average over all possible
        #  permutations, it seems the **m is not needed here.
        coeff = (-2) ** t / double_factorial(
            2 * l3 - 1, 2 * l3 - 2 * t - 1 + 2, device=device
        )

        rule, symmetry = tp_rule_odd(l1, l2, k, t)
        prod = torch.einsum(rule, epsilon, X, Y, *([d] * t))
        prod = symmetrize(prod, symmetry=symmetry)

        out = out + coeff * prod

    out = coeff_D(l1, l2, l3) * out

    return out


def embed_even():
    """
    Embed a natural tensor of rank l3 into the space of rank l1 + l2.

    In some sense, this is the reverse operation of tp_even().

    Returns:

    """


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


# TODO, the symmetry can be simplified to retain fewer terms, if we consider the
#  major symmetry of different deltas
def tp_rule_even(l1: int, l2: int, k: int, t: int) -> tuple[str, str]:
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
            product. e.g. `xxxyyyaa` means the first three indices are symmetric, the
            next three indices are symmetric, and the last two indices are symmetric.
    """

    xy_contracted = letter_index(k + t)
    x_remain = letter_index(l1 - k - t, k + t)
    y_remain = letter_index(l2 - k - t, l1)

    # indices for contracting m I
    # x and y uses l1 + l2 - k - m indices
    delta = double_index(t, l1 + l2 - k - t)
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
        "x" * len(x_remain) + "y" * len(y_remain) + "".join(repeat_double_index(t))
    )

    return rule, symmetry


# TODO, the symmetry can be simplified to retain fewer terms, if we consider the
#  major symmetry of different deltas
def tp_rule_odd(l1: int, l2: int, k: int, t: int) -> tuple[str, str]:
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
    xy_contracted = letter_index(k + t, 3)
    x_remain = letter_index(l1 - 1 - k - t, k + t + 3)
    y_remain = letter_index(l2 - 1 - k - t, l1 + 2)

    # indices for contracting m I
    # x and y uses l1 + l2 - k - m indices
    delta = double_index(t, l1 + l2 - k - t + 1)
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
        + "".join(repeat_double_index(t))
    )

    return rule, symmetry


if __name__ == "__main__":
    from carten.reduce import get_permutations

    rules, symmetry = tp_rule_even(3, 3, 1, 2)
    print("rules", rules)
    print("symmetry", symmetry)
    perms = get_permutations(symmetry)
    print("perms", perms)
