"""Tensor product between two natural tensors.

Ref:
[LP89] "Angular reduction in multiparticle matrix elements" by D. R. Lehman and W. C. Parke.
http://dx.doi.org/10.1063/1.528515
"""
import itertools

import torch
from torch import Tensor

from carten.utils import (
    dij,
    double_factorial,
    double_index,
    eijk,
    factorial,
    get_trace,
    letter_index,
    repeat_double_index,
)


def tp_even(X: Tensor, Y: Tensor, out_rank: int, normalize: str = None) -> Tensor:
    """
    Calculate the tensor product of two natural tensors, when l1 + l2 - l3 is even.

    Args:
        X: A natural tensor of rank l1
        Y: A natural tensor of rank l2
        out_rank: The rank of the output tensor l3
        normalize: The normalization method.
            If `unity`, the output is normalized such that the l3 fold contraction of
            the output tensor with a unit vector yields 1.
            If None, no normalization is applied.

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
        coeff = (-2) ** t / double_factorial(
            2 * l3 - 1, 2 * l3 - 2 * t - 1 + 2, device=device
        )

        rule, symmetry = tp_rule_even(l1, l2, k, t)
        prod = torch.einsum(rule, X, Y, *([d] * t))

        prod = symmetrize(prod, l1, l2, t, k)

        out = out + coeff * prod

    if normalize is None:
        pass
    elif normalize.lower() == "unity":
        out = coeff_C(l1, l2, l3) * out
    else:
        raise ValueError(f"Unknown normalization method: {normalize}")

    return out


def tp_odd(X: Tensor, Y: Tensor, out_rank: int, normalize: str = None) -> Tensor:
    """
    Calculate the tensor product of two natural tensors, when l1 + l2 - l3 is odd.

    Args:
        X: A natural tensor of rank l1
        Y: A natural tensor of rank l2
        out_rank: The rank of the output tensor l3
        normalize: The normalization method.
            If `unity`, the output is normalized such that the l3 fold contraction of
            the output tensor with a unit vector yields 1.
            If None, no normalization is applied.

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
        prod = symmetrize(prod, symmetry=symmetry, mode="sum")

        out = out + coeff * prod

    if normalize is None:
        pass
    elif normalize.lower() == "unity":
        out = coeff_D(l1, l2, l3) * out
    else:
        raise ValueError(f"Unknown normalization method: {normalize}")

    return out


def embed_even():
    """
    Embed a natural tensor of rank l3 into the space of rank l1 + l2.

    In some sense, this is the reverse operation of tp_even().

    Returns:

    """


def coeff_C(l1: int, l2: int, l3: int, device: torch.device = None):
    """Coefficient C for even L.

    The coefficient is obtained such at l3 fold contraction of the output tensor with
    A unit vector yields 1.

    Ref: Eq. 54 of [LP89]
    """
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
    """Coefficient D for odd L.

    The coefficient is obtained such at l3 fold contraction of the output tensor with
    A unit vector yields 1.

    Ref: Eq. 55 of [LP89]
    """
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

    # l1-k-t remaining symmetric indices from x
    # l2-k-t remaining symmetric indices from y
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


def symmetrize(
    Z: Tensor, l1: int, l2: int, t: int, k: int, start_dim: int = 0
) -> Tensor:
    """
    Symmetrize a tensor that is obtained as: Z =  X \odot^(k+t) Y \otimes \delta^t
    where X and Y are symmetric tensors, and \delta is the delta tensor.

    Args:
        Z: the Z tensor
        l1: the rank of the first symmetric tensor X
        l2: the rank of the second symmetric tensor Y
        t: the number of delta tensors
        k: k+t the number of indices contracted between X and Y
        start_dim: the starting dimension to perform the operation. Dimensions before
            `start_dim` will not be used in the operation.


    Returns:
        A symmetrized tensor.
    """

    # Get unique permutations
    permutations = get_permutations(l1, l2, t, k, start_dim)

    # Sum over the permutations
    u = torch.sum(torch.stack([Z.permute(p) for p in permutations]), dim=0)

    return u


# TODO, this can be combined with get_permutations_2 in reduce.py
# We provide symmetry as input.


def get_permutations(
    l1: int, l2: int, t: int, k: int, start_dim: int = 0
) -> list[list[int]]:
    """
    Get unique permutations of the tensor product between two symmetric tensors X and Y
    and delta tensors.

    Z =  X \odot^(k+t) Y \otimes \delta^t

    For example, given X_abce and Y_fghi, when k=0 and t=2, then output tensor Z is a
    rank 8 tensor (l1+l2 - 2(k+t) + 2t).

    We have: Z_cdefghij = X_abcd Y_abef delta_gh delta_ij as one of the terms.
    Will will have C(8, 4) * C(4, 2) * 3 = 70*6*3 =1260 unique permutations.

    First, we consider indices cdef:
        - cd is symmetric
        - ef is symmetric
    So we have 6 permutations for cdef (4 chose 2):
        cdef, cedf, cefd, ecdf, ecfd, efcd
    Second, we consider indices ghij:
        - gh is symmetric
        - ij is symmetric
        - gh and ij are symmetric because both gh and ij are associated with delta
    So, we need to consider both minor and major symmetries for the delta tensors, and
    we have 3 permutations for ghij:
        ghij, gihj, gijh

    All together, we have 1260 unique permutations: choosing 4 indices from 8 to arrange
    the indices of cdef (and the other 4 indices are for ghij); then 6 permutations for
    cdef, and 3 permutations for ghij.

    So, we consider three types of symmetries:
        1. minor symmetry in delta tensors
        2. major symmetry in delta tensors
        3. symmetries in the remaining indices of X and Y

    Args:
        l1: the rank of the first symmetric tensor X
        l2: the rank of the second symmetric tensor Y
        t: the number of delta tensors
        k: k+t the number of indices contracted between X and Y
        start_dim: the starting dimension to perform the operation. Dimensions before
            `start_dim` will not be used in the operation.

    Returns:
        Each inner list contains the permutation indices for symmetrization.
    """

    def canonize(ps: str) -> str:
        """
        Convert a permutation string to its canonical form, such that equivalent
        permutation strings have the same representation.

        This considers:
          - both minor symmetry and major symmetry in delta tensors  (indices a, b, ...)
          - symmetries in the remaining indices of X and Y (indices x, y)

        This is based on first occurrence positions of each letter in the string.
        For example, `baba` and `fefe` are equivalent permutation strings, and
        both will be converted to `0101`.

        Args:
            ps: The permutation string to convert

        Returns:
            The canonical form of the permutation string
        """

        # Do not need to convert x and y to numerical values, because, for example,
        # xxyy and yyxx are different.
        exclude = {"x", "y"}
        return "".join(c if c in exclude else str(ps.index(c)) for c in ps)

    num_x_remain = l1 - k - t
    num_y_remain = l2 - k - t
    assert (
        num_x_remain >= 0 and num_y_remain >= 0
    ), "Both l1 and l2 must be greater than or equal to k + t"

    # Construct the symmetry pattern
    # e.g., xxyyaabb, meaning we have 2 symmetric indices for X, 2 symmetric indices
    # for Y, and 2 pairs of symmetric indices from two deltas.

    x_remain = "x" * (l1 - k - t)
    y_remain = "y" * (l2 - k - t)
    delta = "".join(repeat_double_index(t))

    symmetry = f"{x_remain}{y_remain}{delta}"

    all_perms = itertools.permutations(range(start_dim, start_dim + len(symmetry)))

    prefix = list(range(start_dim))
    unique_perms = []
    unique_canonical_forms = set()

    # Filter permutations based on contraction pattern
    for perm in all_perms:
        perm_string = "".join(symmetry[i - start_dim] for i in perm)

        canonical_form = canonize(perm_string)
        if canonical_form not in unique_canonical_forms:
            unique_perms.append(prefix + list(perm))
            unique_canonical_forms.add(canonical_form)

    return unique_perms


if __name__ == "__main__":
    # from carten.reduce import get_permutations
    #
    # rules, symmetry = tp_rule_even(3, 3, 1, 2)
    # print("rules", rules)
    # print("symmetry", symmetry)
    # perms = get_permutations(symmetry)
    # print("perms", perms)
    from carten.reduce import symmetrize_and_remove_trace
    from carten.utils import is_symmetric, is_symmetric_traceless, is_traceless

    torch.manual_seed(0)
    T2 = torch.randn(3, 3)
    T3 = torch.randn(3, 3, 3)

    NT2 = symmetrize_and_remove_trace(T2)
    NT3 = symmetrize_and_remove_trace(T3)
    assert is_symmetric_traceless(NT2), "NT2 is not symmetric traceless"
    assert is_symmetric_traceless(NT3), "NT3 is not symmetric traceless"

    out = tp_even(NT2, NT2, 4)
    # out = tp_odd(NT2, NT3, 4)
    print(out)
    print("trace 0, 1", get_trace(out, 0, 1))
    assert is_symmetric(out), "out is not symmetric"
    assert is_traceless(out, atol=1e-5), "out is not traceless"

    # out2 = symmetrize_and_remove_trace(torch.einsum("ab,cd", T2, T2))
    # assert is_symmetric(out2), "out2 is not symmetric"
    # assert is_traceless(out2, atol=1e-5), "out2 is not traceless"
