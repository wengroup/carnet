"""Natural tensors constructed from unit vectors."""

import torch
from torch import Tensor

from carten.core.symmetrize import get_permutations_delta, symmetrize_via_permutation
from carten.core.utils import (
    dij,
    double_factorial,
    double_index,
    factorial,
    letter_index,
    repeat_double_index,
)


def get_nt_from_vector(
    a: Tensor, l: int, normalize: str = "unity", flatten: bool = False
) -> Tensor:
    """
    Create a natural tensor from a unit vector.

    X = C \sum_{d=0}^D (-1)^d \frac{(2l-2d-1)!!}{(2l-1)!!}
    \{ \hat{\bm a}^{\otimes^{l-2d}}\otimes \bm I^{\otimes d} \},

    where $D = n/2$ for even $n$ and $D = (n-1)/2$ for odd $n$.

    The constant $C$ is a normalization factor.

    Args:
        a: The unit vector(s). Shape(..., 3), where the last dimension is the vector,
            and the rest are batch dimensions.
        l: Rank of the natural tensor to create.
        normalize: Normalization type.
            If `unity`, $C = \frac{(2l-1)!!}{l!}$ is used for normalization.
            In this case, an l-contraction between the output natural tensor and an
            arbitrary unit vector `b` is equal to the Legendre polynomial of the angle
            between `a` and `b`. Namely: $out \odot^l b^{\otimes^l} = P_l(a \cdot b)$.
            If `b` is chosen to be `a`, the l-contraction between the output natural
            and `a` is equal to 1, i.e. $out \odot^l a^{\otimes^l} = 1$.
            If `none`, no normalization is applied, i.e. $C = 1$.
        flatten: Whether to flatten the tensor dims. If `False`, the output tensor will
            have shape (..., 3, 3, ..., 3), where the number of 3s is `l`. If `True`,
            the output tensor will have shape (..., 3**l).

    Returns:
            The rank-l natural tensor constructed from the unit vector.
    """
    # TODO we can force to normalize `a` as a unit vector

    # For rank-0, return scalar 1. For rank-1, return the unit vector itself.
    batch_dims = a.shape[:-1]

    if l == 0:
        return torch.atleast_1d(torch.ones(batch_dims, dtype=a.dtype, device=a.device))

    elif l == 1:
        return a

    delta = dij(a.device)

    D = l // 2

    out = torch.zeros([3] * l, dtype=a.dtype, device=a.device)
    coeff = 1
    for d in range(D + 1):
        rule, symmetry, delta_indices = get_nt_from_vector_rule(l, d)

        # When l == 2*d, we have only deltas, and we create a placeholder of 1 to deal
        # with the batch dimensions in the vector `a`.
        if l == 2 * d:
            all_a = [torch.ones(batch_dims, dtype=a.dtype, device=a.device)]
        else:
            all_a = [a] * (l - 2 * d)

        # Get one tensor product
        prod = torch.einsum(rule, *all_a, *([delta] * d))

        # Symmetrize by summing over all unique permutations
        perms = get_permutations_delta(symmetry, delta_indices, start_dim=a.ndim - 1)
        prod = symmetrize_via_permutation(prod, perms, mode="sum")

        out = out + coeff * prod

        # Update the coefficient
        # coeff = (-1) ** d / double_factorial(2 * l - 1, 2 * l - 2 * d - 1 + 2)
        coeff = -coeff / (2 * l - 2 * d - 1)

    if normalize == "unity":
        out = double_factorial(2 * l - 1) / factorial(l) * out
    elif normalize == "none":
        pass
    else:
        raise ValueError(f"Unknown normalization method: {normalize}")

    if flatten:
        return out.view(batch_dims + (-1,))
    else:
        return out


def get_polyadics_from_vector(a: Tensor, L: int, normalize="unity"):
    """
    Create polyadic tensors from a unit vector.

    A polyadic tensor of rank L from a unit vector is defined as:
    $a \otimes a \otimes ... \otimes a$,  # a total of L a.
    It can be decomposed to natural tensors of unit vector of rank 0, 1, and up to L.

    This function gets all the natural tensors and concatenates them at the last
    dimension.

    Args:
        a: The unit vector(s). Shape(..., 3), where the last dimension is the vector,
            and the rest are batch dimensions.
        L: Maximum rank of the natural tensors to create.
        normalize: Normalization type. See `get_nt_from_vector()`.

    Returns:
        The feature tensor of the unit vector. Shape (..., T), where
        T = \sum_{l=0}^L = ((L+1)**2 -1)/2.
    """
    feature_tensors = []
    for l in range(L + 1):
        feature_tensors.append(get_nt_from_vector(a, l, normalize, flatten=True))

    return torch.cat(feature_tensors, dim=-1)


def get_nt_from_vector_rule(l: int, d: int) -> tuple[str, str, str]:
    """
    Get the rule for creating a rank-l natural tensor from a unit vector.

    Args:
        l: Rank of the natural tensor to create.
        d: Number of deltas.

    Returns:
        rule: The rule for creating a rank-l natural tensor from a unit vector.
        symmetry: The symmetry of the rule.
        delta_indices: The delta indices.
    """
    a = letter_index(l - 2 * d)
    a_left = "..." + ",...".join(a)  # the first `...` is for batch dimensions
    a_right = "..." + a  # the first `...` is for batch dimensions

    delta = double_index(d, start=l - 2 * d)
    delta_right = "".join(delta)
    delta_left = ",".join(delta) if delta else ""
    if a_left and delta_left:
        delta_left = "," + delta_left

    rule = f"{a_left}{delta_left}->{a_right}{delta_right}"
    symmetry = "x" * (l - 2 * d) + "".join(repeat_double_index(d))
    delta_indices = letter_index(d)

    return rule, symmetry, delta_indices
