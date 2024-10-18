"""Tensor product between two natural tensors.

This uses the explicit formula given in
"Angular reduction in multiparticle matrix elements" by D. R. Lehman and W. C. Parke.
http://dx.doi.org/10.1063/1.528515
"""
import itertools

import torch

from carten.natural_tensor import NaturalTensor
from carten.utils import dij, letter_index


def tp_even(S: NaturalTensor, T: NaturalTensor, out_rank: int) -> NaturalTensor:
    """
    Calculate the tensor product of two natural tensors, when l1 + l2 - l3 is even.

    Args:
        S: A natural tensor of rank l1
        T: A natural tensor of rank l2
        out_rank: The rank of the output tensor l3

    Returns:
        A natural tensor of rank l3
    """
    assert (S.dim() + T.dim() - out_rank) % 2 == 0, "l1 + l2 - l3 must be even"

    l1 = S.shape[0]
    l2 = T.shape[0]
    l3 = out_rank
    dtype = S.dtype
    device = S.device

    k = (l1 + l2 - l3) // 2
    out = torch.zeros(*[3] * out_rank, dtype=dtype, device=device)

    d = dij(device)

    for m in range(min(l1, l2) - k + 1):
        coeff = (
            (-2) ** m
            * double_factorial(2 * l3 - 2 * m - 1, device)
            / double_factorial(2 * l3 - 1, device)
        )

        rule, symmetry = tp_rule_even(l1, l2, k, m)
        prod = torch.einsum(rule, S, T, *[d] * m)
        prod = symmetrize(prod, symmetry)

        out = out + coeff * prod

    out = coeff_C(l1, l2, l3) * out

    return out


def coeff_C(l1: int, l2: int, l3: int, device: torch.device = None):
    """Coefficient C"""
    L = l1 + l2 + l3
    L1 = L - 2 * l1 - 1
    L2 = L - 2 * l2 - 1
    L3 = L - 2 * l3 - 1

    return (
        factorial(l1, device)
        * factorial(l2, device)
        * double_factorial(2 * l3 - 1, device)
        * factorial((L1 + 1) // 2, device)
        * factorial((L2 + 1) // 2, device)
        / factorial(l3, device)
        / double_factorial(L1, device)
        / double_factorial(L2, device)
        / double_factorial(L3, device)
        / factorial(L // 2, device)
    )


def factorial(n: int, device: torch.device = None):
    """
    Get the factorial of a number.
    """
    return torch.prod(torch.arange(1, n + 1, device=device))


def double_factorial(n: int, device: torch.device = None):
    """
    Get the double factorial of a number.
    """
    if n == 0 or n == 1:
        return torch.tensor(1, device=device)
    elif n % 2 == 0:
        return torch.prod(torch.arange(2, n, step=2, device=device))
    else:
        return torch.prod(torch.arange(1, n, step=2, device=device))


def tp_rule_even(l1: int, l2: int, k: int, m: int) -> tuple[str, str]:
    """Get the einsum rule when l1 + l2 + l3 is even.

    x_l1 \otimes^{k+m} x_l2 \otimes I ^{\otimes^m}

    After contraction, the resultant tensor will have l1-k-m indices from x, and these
    indices are still symmetric. Similarly, the resultant tensor will have l2-k-m
    symmetric indices from y. It will have 2*m indices from I. Each two indices from I
    are symmetric.

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
    delta = letter_index(2 * m, l1 + l2 - k - m)
    delta = [delta[i : i + 2] for i in range(0, 2 * m, 2)]
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
        "a" * len(x_remain)
        + "b" * len(y_remain)
        + "".join(c * 2 for c in letter_index(m, start=2))  # start: c
    )

    return rule, symmetry


def symmetrize(t: torch.Tensor, symmetry: str = None) -> torch.Tensor:
    """
    Symmetrize a tensor.

    The symmetrization is done by averaging over unique permutations of the indices,
    considering the symmetry of the indices.

    Args:
        t: The tensor to symmetrize
        symmetry: A string that describes the symmetry of the indices. For example,
            `abba` means the first and the fourth indices are symmetric, and the
            second and the third indices are symmetric. Default is None, which means
            there is no symmetry between the indices.

    Returns:
        The symmetrized tensor.
    """
    if symmetry is None:
        permutations = itertools.permutations(range(t.dim()))
    else:
        permutations = get_permutations(symmetry)

    sym_t = torch.mean(torch.stack([t.permute(p) for p in permutations]), dim=0)

    return sym_t


def get_permutations(symmetry: str) -> list[tuple[int, ...]]:
    """
    Get the unique permutations of the indices for symmetrizing a tensor.

    This works for the case where part or all of the indices are symmetric.

    Args:
        symmetry: A string that describes the symmetry of the indices. For example,
            `abba` means the first and the fourth indices are symmetric, and the
            second and the third indices are symmetric.

    Example:
        >>> get_permutations('abba')
        [(0, 1, 2, 3),  # abba
         (0, 1, 3, 2),  # abab
         (0, 3, 1, 2),  # aabb
         (1, 0, 2, 3),  # baba
         (1, 0, 3, 2),  # baab
         (1, 2, 0, 3)]  # bbaa

    Returns:
        Each inner list contains the permutation indices for symmetrization.
    """
    # TODO, this is a generalization of get_sym_rule_2 and get_sym_rule_3 in tensor_product1.py
    #  Can we merge them?

    all_perms = itertools.permutations(range(len(symmetry)))

    unique_perms = []
    unique_perm_string = set()
    for perm in all_perms:
        perm_string = "".join(symmetry[i] for i in perm)

        if perm_string not in unique_perm_string:
            unique_perms.append(perm)
            unique_perm_string.add(perm_string)

    return unique_perms
