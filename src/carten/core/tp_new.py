"""Tensor product between two natural tensors.

Batching and feature dimensions of the tensors are supported.

Ref:
[LP89] "Angular reduction in multiparticle matrix elements" by D. R. Lehman and W. C. Parke.
http://dx.doi.org/10.1063/1.528515
"""

from typing import Optional

import torch
from line_profiler import profile
from natt.evaluate import evaluate_tensors
from natt.GHS import get_G_H_S_of_j_natural
from natt.symmetrize import get_permutations_delta, symmetrize_via_permutation
from natt.utils import dij, double_index, eijk, letter_index, repeat_double_index
from torch import Tensor

from carten.core.utils import double_factorial, factorial

# CACHE to speed up calculation, they will be filled when the functions are called
TP_EVEN_G_CACHE = {}
TP_EVEN_XY_RULE_CACHE = {}
TP_EVEN_Z_RULE_CACHE = {}

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

    assert (l1 + l2 - l3) % 2 == 0, "l1 + l2 - l3 must be even"

    leading_dims = X.shape[:-1]  # including the feature dimension
    dtype = X.dtype
    device = X.device

    k = (l1 + l2 - l3) // 2

    d = dij(device)

    if l1 == 0 or l2 == 0:
        # Scalars are simple
        Z = X * Y
    else:

        # Expand the tensors dims: (..., 3^l1) -> (..., 3, 3, ..., 3)
        X = X.view(leading_dims + (3,) * l1)
        Y = Y.view(leading_dims + (3,) * l2)

        global TP_EVEN_G_CACHE
        global TP_EVEN_XY_RULE_CACHE
        global TP_EVEN_Z_RULE_CACHE
        if (l1, l2, l3) not in TP_EVEN_G_CACHE:
            G, _, _, _, _ = get_G_H_S_of_j_natural(l1, l2, l3)
            G = evaluate_tensors(G, mode="G")
            G = G.to(X.device)

            X_indices = letter_index(l1)
            Y_indices = letter_index(l2, start=l1)
            XY_indices = X_indices + Y_indices
            XY_rule = f"...{X_indices},...{Y_indices}->...{XY_indices}"

            Z_indices = letter_index(l3, upper_case=True)
            Z_rule = f"...{Z_indices}{XY_indices},...{XY_indices}->...{Z_indices}"

            # cache
            TP_EVEN_G_CACHE[(l1, l2, l3)] = G
            TP_EVEN_XY_RULE_CACHE[(l1, l2, l3)] = XY_rule
            TP_EVEN_Z_RULE_CACHE[(l1, l2, l3)] = Z_rule

        else:
            G = TP_EVEN_G_CACHE[(l1, l2, l3)]
            XY_rule = TP_EVEN_XY_RULE_CACHE[(l1, l2, l3)]
            Z_rule = TP_EVEN_Z_RULE_CACHE[(l1, l2, l3)]

        # TODO, these should be combined
        XY = torch.einsum(XY_rule, X, Y)
        Z = torch.einsum(Z_rule, G, XY)

        # Combine the tensor dims: (..., 3, 3, ..., 3) -> (..., 3^l3)
        Z = Z.view(leading_dims + (-1,))

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
        raise ValueError(f"Unknown normalization method: {normalize}")

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

    # Combine the tensor dims: (..., 3, 3, ..., 3) -> (..., 3^l3)
    Z = Z.view(leading_dims + (-1,))

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

    return Z


def coeff_C(l1: int, l2: int, l3: int, device: Optional[torch.device] = None):
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


def coeff_D(l1: int, l2: int, l3: int, device: Optional[torch.device] = None):
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

    In total, the resultant tensor will have l3 = l1 + l2 - 2(k + t) tensor indices.

    Returns:
        rule: The einsum rule for the tensor product
        symmetry: The symmetry information of the resultant tensor after the tensor.
            product. e.g. `xxxyyyaa` means the first three indices are symmetric, the
            next three indices are symmetric, and the last two indices are symmetric.
        delta_indices: The indices for the delta tensors.
    """

    # indices that are contracted
    xy_contracted = letter_index(k + t)

    # indices that are not contracted
    x_remain = letter_index(l1 - k - t, k + t)
    y_remain = letter_index(l2 - k - t, l1)

    # indices for contracting t of I
    delta = double_index(t, upper_case=True)
    delta_left = "," + ",".join(delta) if delta else ""
    delta_right = "".join(delta)

    rule = (
        f"...{xy_contracted}{x_remain},"
        f"...{xy_contracted}{y_remain}"
        f"{delta_left}"
        f"->...{x_remain}{y_remain}{delta_right}"
    )

    # l1-k-t remaining symmetric indices from x
    # l2-k-t remaining symmetric indices from y
    # 2t indices from all deltas. Each delta has 2 symmetric indices.
    # TorchScript does not allow string multiplication, so we need to use `join`
    symmetry = (
        "".join(["a"] * len(x_remain))
        + "".join(["b"] * len(y_remain))
        + "".join(repeat_double_index(t, upper_case=True))
    )
    delta_indices = letter_index(t, upper_case=True)

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
    # example: epsilon_uvw x_vabc  y_wabd I_AB -> ucdAB

    xy_contracted = letter_index(k + t)
    x_remain = letter_index(l1 - 1 - k - t, k + t)
    y_remain = letter_index(l2 - 1 - k - t, l1 - 1)

    # indices for contracting t of I
    delta = double_index(t, upper_case=True)
    delta_left = "," + ",".join(delta) if delta else ""
    delta_right = "".join(delta)

    # The ... are for the batch dimensions
    rule = (
        f"uvw,"  # indices for epsilon
        f"...v{xy_contracted}{x_remain},"
        f"...w{xy_contracted}{y_remain}"
        f"{delta_left}"
        f"->...u{x_remain}{y_remain}{delta_right}"
    )

    # 1 index from epsilon
    # l1-1-k-t remaining symmetric indices from x
    # l2-1-k-t remaining symmetric indices from y
    # 2t indices from all deltas. Each delta has 2 symmetric indices.
    symmetry = (
        "a"
        + "".join(["b"] * len(x_remain))
        + "".join(["c"] * len(y_remain))
        + "".join(repeat_double_index(t, upper_case=True))
    )
    delta_indices = letter_index(t, upper_case=True)

    return rule, symmetry, delta_indices
