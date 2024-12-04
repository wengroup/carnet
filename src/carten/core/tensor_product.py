"""Tensor product between two natural tensors.

Ref:
[LP89] "Angular reduction in multiparticle matrix elements" by D. R. Lehman and W. C. Parke.
http://dx.doi.org/10.1063/1.528515
"""
import torch
from torch import Tensor

from carten.core.permute import get_permutations_delta
from carten.core.utils import (
    dij,
    double_factorial,
    double_index,
    eijk,
    factorial,
    letter_index,
    repeat_double_index,
)


def tp_even(X: Tensor, Y: Tensor, out_rank: int, normalize: str = "unity") -> Tensor:
    """
    Calculate the tensor product of two natural tensors, when l1 + l2 - l3 is even.

    Args:
        X: A natural tensor of rank l1
        Y: A natural tensor of rank l2
        out_rank: The rank of the output tensor l3
        normalize: The normalization method.
            If `unity`, the output is normalized such that the l3 fold contraction of
            the output tensor with a unit vector yields 1.
            If `none`, no normalization is applied.

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
        # TODO, coeff might be simplified by recursion, see unit_vector.py
        coeff = (-2) ** t / double_factorial(
            2 * l3 - 1, 2 * l3 - 2 * t - 1 + 2, device=device
        )

        rule, symmetry, delta_indices = get_tp_even_rule(l1, l2, k, t)

        # Get one tensor product
        prod = torch.einsum(rule, X, Y, *([d] * t))

        # Symmetrize by summing over all unique permutations
        perms = get_permutations_delta(symmetry, delta_indices)
        prod = torch.sum(torch.stack([prod.permute(p) for p in perms]), dim=0)

        out = out + coeff * prod

    if normalize == "unity":
        out = coeff_C(l1, l2, l3) * out
    elif normalize == "none":
        pass
    else:
        raise ValueError(f"Unknown normalization method: {normalize}")

    return out


def tp_odd(X: Tensor, Y: Tensor, out_rank: int, normalize: str = "unity") -> Tensor:
    """
    Calculate the tensor product of two natural tensors, when l1 + l2 - l3 is odd.

    Args:
        X: A natural tensor of rank l1
        Y: A natural tensor of rank l2
        out_rank: The rank of the output tensor l3
        normalize: The normalization method.
            If `unity`, the output is normalized such that the l3 fold contraction of
            the output tensor with a unit vector yields 1.
            If `none`, no normalization is applied.

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
        coeff = (-2) ** t / double_factorial(
            2 * l3 - 1, 2 * l3 - 2 * t - 1 + 2, device=device
        )

        rule, symmetry, delta_indices = get_tp_odd_rule(l1, l2, k, t)

        # get one tensor product
        prod = torch.einsum(rule, epsilon, X, Y, *([d] * t))

        # get all tensor products by symmetrizing the one tensor product
        perms = get_permutations_delta(symmetry, delta_indices)
        prod = torch.sum(torch.stack([prod.permute(p) for p in perms]), dim=0)

        out = out + coeff * prod

    if normalize == "unity":
        out = coeff_D(l1, l2, l3) * out
    elif normalize == "none":
        pass
    else:
        raise ValueError(f"Unknown normalization method: {normalize}")

    return out


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


def get_tp_even_rule(l1: int, l2: int, k: int, t: int) -> tuple[str, str, str]:
    """
    Get the einsum rule when l1 + l2 - l3 is even.

    x_l1 \odot^{k+t} x_l2 \otimes I ^{\otimes^m}

    After contraction, the resultant tensor will have l1-k-t indices from x, and these
    indices are still symmetric. Similarly, the resultant tensor will have l2-k-t
    symmetric indices from y. It will have 2*t indices from I. Each two indices from I
    are symmetric.

    In total, the resultant tensor will have l3 = l1 + l2 - 2(k + t) indices.

    Returns:
        rule: The einsum rule for the tensor product
        symmetry: The symmetry information of the resultant tensor after the tensor
            product. e.g. `xxxyyyaa` means the first three indices are symmetric, the
            next three indices are symmetric, and the last two indices are symmetric.
        delta_indices: The indices for the delta tensors.
    """

    xy_contracted = letter_index(k + t)
    x_remain = letter_index(l1 - k - t, k + t)
    y_remain = letter_index(l2 - k - t, l1)

    # indices for contracting t of I
    # x and y uses l1 + l2 - k - t indices
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

    delta_indices = letter_index(t)

    return rule, symmetry, delta_indices


def get_tp_odd_rule(l1: int, l2: int, k: int, t: int) -> tuple[str, str, str]:
    """
    Get the einsum rule when l1 + l2 - l3 is odd.

    epsilon : x_l1 \odot^{k+t} x_l2 \otimes I ^{\otimes^t}

    epsilon is the Levi-Civita symbol. It contracts away one index from x and one index
    from y. So, after contraction, the resultant tensor will have 1 index from epsilon.
    After contraction, the resultant tensor will have l1-1-k-t indices from x, and these
    indices are still symmetric. Similarly, the resultant tensor will have l2-1-k-t
    symmetric indices from y. It will have 2*m indices from I. Each two indices from I
    are symmetric.

    In total, the resultant tensor will have l3 = l1 + l2 - 1 - 2(k + t) indices.


    Returns:
        rule: The einsum rule for the tensor product
        symmetry: The symmetry information of the resultant tensor after the tensor
            product. e.g. `aabb` means the first two indices are symmetric, and the last
            two indices are symmetric.
        delta_indices: The indices for the delta tensors.
    """
    # example: epsilon_abc x_bpqr  y_cpqs I_tu -> arstu

    # epsilon_remain = "a"
    # epsilon_contracted = "bc"
    xy_contracted = letter_index(k + t, 3)
    x_remain = letter_index(l1 - 1 - k - t, k + t + 3)
    y_remain = letter_index(l2 - 1 - k - t, l1 + 2)

    # indices for contracting t of I
    # x and y uses l1 + l2 - k - t + 1 indices
    delta = double_index(t, l1 + l2 - k - t + 1)
    delta_left = "," + ",".join(delta) if delta else ""
    delta_right = "".join(delta)

    rule = (
        f"abc,b{xy_contracted}{x_remain},c{xy_contracted}{y_remain}{delta_left}"
        f"->a{x_remain}{y_remain}{delta_right}"
    )

    # 1 index from epsilon
    # l1-1-k-t remaining symmetric indices from x
    # l2-1-k-t remaining symmetric indices from y
    # 2t indices from all deltas. Each delta has 2 symmetric indices.
    symmetry = (
        "x"
        + "y" * len(x_remain)
        + "z" * len(y_remain)
        + "".join(repeat_double_index(t))
    )

    delta_indices = letter_index(t)

    return rule, symmetry, delta_indices


if __name__ == "__main__":
    from carten.core.reduce import symmetrize_and_remove_trace
    from carten.core.utils import is_symmetric_traceless

    torch.manual_seed(0)
    T2 = torch.randn(3, 3)
    T3 = torch.randn(3, 3, 3)

    NT2 = symmetrize_and_remove_trace(T2)
    NT3 = symmetrize_and_remove_trace(T3)
    assert is_symmetric_traceless(NT2), "NT2 is not symmetric traceless"
    assert is_symmetric_traceless(NT3), "NT3 is not symmetric traceless"

    out = tp_even(NT2, NT2, 4, normalize="unity")
    assert is_symmetric_traceless(out, atol=1e-5), "out is not symmetric traceless"

    out = tp_odd(NT2, NT3, 4, normalize="unity")
    assert is_symmetric_traceless(out, atol=1e-5), "out is not symmetric traceless"
