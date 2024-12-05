"""Tensor product between two natural tensors.

Batching and multiplicity of the tensors are supported.

Ref:
[LP89] "Angular reduction in multiparticle matrix elements" by D. R. Lehman and W. C. Parke.
http://dx.doi.org/10.1063/1.528515
"""
import torch
from torch import Tensor

from carten.core.symmetrize import get_permutations_delta, symmetrize_via_permutation
from carten.core.utils import (
    dij,
    double_factorial,
    double_index,
    eijk,
    factorial,
    letter_index,
    repeat_double_index,
)


def tp_even(
    X: Tensor, Y: Tensor, l1: int, l2: int, l3: int, normalize: str = "unity"
) -> Tensor:
    """
    Calculate the tensor product of two natural tensors, when l1 + l2 - l3 is even.

    This functions considers batching and multiplicity of the natural tensors.
    If your input `X` and `Y` are natural tensors without batching and multiplicity,
    use `tp_even_simple()` instead.

    Note:
        The multiplicity of the output tensor is the product of the multiplicities of
        the two input tensors.

    Args:
        X: A natural tensor of rank l1. Shape: (..., m1, 3, 3, ..., 3), where the first
            ... indices are the batch dimensions, m1 is the multiplicity of X, and the
            3, 3, ..., 3 (a total number of l1) are the tensor indices.
        Y: A natural tensor of rank l2. Shape: (..., m2, 3, 3, ..., 3), where the first
            ... indices are the batch dimensions, m2 is the multiplicity of Y, and the
            3, 3, ..., 3 (a total number of l2) are the tensor indices. The batch dims
            of X and Y must be the same.
        l1: The rank of the first tensor X.
        l2: The rank of the second tensor Y.
        l3: The rank of the output tensor Z.
        normalize: The normalization method.
            If `unity`, the output is normalized such that the l3 fold contraction of
            the output tensor with a unit vector yields 1.
            If `none`, no normalization is applied.

    Returns:
        A natural tensor of rank l3. Shape: (..., m3, 3, 3, ..., 3), where the
        batching dimensions are the same as the input tensors X and Y, the multiplicity
        m3 is the product of the multiplicities of X and Y (i.e. m1*m3), and the
        3, 3, ..., 3 (a total number of l3) are the tensor indices.
    """

    assert (l1 + l2 - l3) % 2 == 0, "l1 + l2 - l3 must be even"

    batch_dims = X.shape[: -l1 - 1]
    m1 = X.shape[-l1 - 1]
    m2 = Y.shape[-l2 - 1]
    m3 = m1 * m2

    dtype = X.dtype
    device = X.device

    k = (l1 + l2 - l3) // 2

    d = dij(device)

    # Create the output tensor
    out_shape = batch_dims + (m3,) + (3,) * l3
    Z = torch.zeros(out_shape, dtype=dtype, device=device)

    for t in range(min(l1, l2) - k + 1):
        coeff = (-2) ** t / double_factorial(
            2 * l3 - 1, 2 * l3 - 2 * t - 1 + 2, device=device
        )

        rule, symmetry, delta_indices = get_tp_even_rule(l1, l2, k, t)

        # Get one tensor product
        prod = torch.einsum(rule, X, Y, *([d] * t))

        # Combine the m1, m2 dimensions into a single m3 dimension
        prod = prod.reshape(out_shape)

        # Symmetrize by summing over all unique permutations
        start_dim = len(batch_dims) + 1  # +1 for the multiplicity dimension
        perms = get_permutations_delta(symmetry, delta_indices, start_dim)
        prod = symmetrize_via_permutation(prod, perms, mode="sum")

        Z = Z + coeff * prod

    if normalize == "unity":
        Z = coeff_C(l1, l2, l3) * Z
    elif normalize == "none":
        pass
    else:
        raise ValueError(f"Unknown normalization method: {normalize}")

    return Z


def tp_odd(
    X: Tensor, Y: Tensor, l1: int, l2: int, l3: int, normalize: str = "unity"
) -> Tensor:
    """
    Calculate the tensor product of two natural tensors, when l1 + l2 - l3 is odd.

    This functions considers batching and multiplicity of the natural tensors.
    If your input `X` and `Y` are natural tensors without batching and multiplicity,
    use `tp_odd_simple()` instead.

    Note:
        The multiplicity of the output tensor is the product of the multiplicities of
        the two input tensors.

    Args:
        X: A natural tensor of rank l1. Shape: (..., m1, 3, 3, ..., 3), where the first
            ... indices are the batch dimensions, m1 is the multiplicity of X, and the
            3, 3, ..., 3 (a total number of l1) are the tensor indices.
        Y: A natural tensor of rank l2. Shape: (..., m2, 3, 3, ..., 3), where the first
            ... indices are the batch dimensions, m2 is the multiplicity of Y, and the
            3, 3, ..., 3 (a total number of l2) are the tensor indices. The batch dims
            of X and Y must be the same.
        l1: The rank of the first tensor X.
        l2: The rank of the second tensor Y.
        l3: The rank of the output tensor Z.
        normalize: The normalization method.
            If `unity`, the output is normalized such that the l3 fold contraction of
            the output tensor with a unit vector yields 1.
            If `none`, no normalization is applied.

    Returns:
        A natural tensor of rank l3. Shape: (..., m3, 3, 3, ..., 3), where the
        batching dimensions are the same as the input tensors X and Y, the multiplicity
        m3 is the product of the multiplicities of X and Y (i.e. m1*m3), and the
        3, 3, ..., 3 (a total number of l3) are the tensor indices.
    """

    assert (l1 + l2 - l3) % 2 == 1, "l1 + l2 - l3 must be odd"

    batch_dims = X.shape[: -l1 - 1]
    m1 = X.shape[-l1 - 1]
    m2 = Y.shape[-l2 - 1]
    m3 = m1 * m2

    dtype = X.dtype
    device = X.device

    k = (l1 + l2 - l3 - 1) // 2

    d = dij(device)
    epsilon = eijk(device)

    # Create the output tensor
    out_shape = batch_dims + (m3,) + (3,) * l3
    Z = torch.zeros(out_shape, dtype=dtype, device=device)

    for t in range(min(l1, l2) - k):
        coeff = (-2) ** t / double_factorial(
            2 * l3 - 1, 2 * l3 - 2 * t - 1 + 2, device=device
        )

        rule, symmetry, delta_indices = get_tp_odd_rule(l1, l2, k, t)

        # Get one tensor product
        prod = torch.einsum(rule, epsilon, X, Y, *([d] * t))

        # Combine the m1, m2 dimensions into a single m3 dimension
        prod = prod.reshape(out_shape)

        # Symmetrize by summing over all unique permutations
        start_dim = len(batch_dims) + 1  # +1 for the multiplicity dimension
        perms = get_permutations_delta(symmetry, delta_indices, start_dim)
        prod = symmetrize_via_permutation(prod, perms, mode="sum")

        Z = Z + coeff * prod

    if normalize == "unity":
        Z = coeff_D(l1, l2, l3) * Z
    elif normalize == "none":
        pass
    else:
        raise ValueError(f"Unknown normalization method: {normalize}")

    return Z


def tp_even_simple(X: Tensor, Y: Tensor, l3: int, normalize: str = "unity") -> Tensor:
    """
    The same as `tp_even()`, but without batching and multiplicity.

    Args:
        X: A natural tensor of rank l1. Shape: (3, 3, ..., 3), a total of l1 3's.
        Y: A natural tensor of rank l2. Shape: (3, 3, ..., 3), a total of l2 3's.
        l3: The rank of the output tensor.
        normalize: The normalization method.

    Returns:
        A natural tensor of rank l3. Shape: (3, 3, ..., 3), a total of l3 3's.
    """
    l1 = X.ndim
    l2 = Y.ndim

    # Add a single multiplicity dimension
    X = X.unsqueeze(0)
    Y = Y.unsqueeze(0)

    Z = tp_even(X, Y, l1, l2, l3, normalize)

    # Remove the multiplicity dimension
    Z = Z.squeeze(0)

    return Z


def tp_odd_simple(X: Tensor, Y: Tensor, l3: int, normalize: str = "unity") -> Tensor:
    """
    The same as `tp_odd()`, but without batching and multiplicity.

    Args:
        X: A natural tensor of rank l1. Shape: (3, 3, ..., 3), a total of l1 3's.
        Y: A natural tensor of rank l2. Shape: (3, 3, ..., 3), a total of l2 3's.
        l1: The rank of the first tensor X.
        l2: The rank of the second tensor Y.
        l3: The rank of the output tensor.
        normalize: The normalization method.

    Returns:
        A natural tensor of rank l3. Shape: (3, 3, ..., 3), a total of l3 3's.
    """

    l1 = X.ndim
    l2 = Y.ndim

    # Add a single multiplicity dimension
    X = X.unsqueeze(0)
    Y = Y.unsqueeze(0)

    Z = tp_odd(X, Y, l1, l2, l3, normalize)

    # Remove the multiplicity dimension
    Z = Z.squeeze(0)

    return Z


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

    In total, the resultant tensor will have l3 = l1 + l2 - 2(k + t) tensor indices.

    Returns:
        rule: The einsum rule for the tensor product
        symmetry: The symmetry information of the resultant tensor after the tensor
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

    # indices for multiplicity
    x_m = "x"
    y_m = "y"

    # The ... are for the batch dimensions
    rule = (
        f"...{x_m}{xy_contracted}{x_remain},"
        f"...{y_m}{xy_contracted}{y_remain}"
        f"{delta_left}"
        f"->...{x_m}{y_m}{x_remain}{y_remain}{delta_right}"
    )

    # l1-k-t remaining symmetric indices from x
    # l2-k-t remaining symmetric indices from y
    # 2t indices from all deltas. Each delta has 2 symmetric indices.
    symmetry = (
        "a" * len(x_remain)
        + "b" * len(y_remain)
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
    # example: epsilon_abc x_bpqr  y_cpqs I_tu -> arstu

    xy_contracted = letter_index(k + t)
    x_remain = letter_index(l1 - 1 - k - t, k + t)
    y_remain = letter_index(l2 - 1 - k - t, l1 - 1)

    # indices for contracting t of I
    delta = double_index(t, upper_case=True)
    delta_left = "," + ",".join(delta) if delta else ""
    delta_right = "".join(delta)

    # indices for multiplicity
    x_m = "x"
    y_m = "y"

    # The ... are for the batch dimensions
    rule = (
        f"uvw,"  # indices for epsilon
        f"...{x_m}v{xy_contracted}{x_remain},"
        f"...{y_m}w{xy_contracted}{y_remain}"
        f"{delta_left}"
        f"->...{x_m}{y_m}u{x_remain}{y_remain}{delta_right}"
    )

    # 1 index from epsilon
    # l1-1-k-t remaining symmetric indices from x
    # l2-1-k-t remaining symmetric indices from y
    # 2t indices from all deltas. Each delta has 2 symmetric indices.
    symmetry = (
        "a"
        + "b" * len(x_remain)
        + "c" * len(y_remain)
        + "".join(repeat_double_index(t, upper_case=True))
    )
    delta_indices = letter_index(t, upper_case=True)

    return rule, symmetry, delta_indices
