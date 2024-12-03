"""Natural tensors constructed from unit vectors."""

import torch
from torch import Tensor

from carten.core.permute import get_permutations_delta
from carten.core.utils import (
    dij,
    double_factorial,
    double_index,
    factorial,
    letter_index,
    repeat_double_index,
)


def get_nt_from_vector(a: Tensor, n: int, normalize: str = "unity") -> Tensor:
    """
    Create a rank-n natural tensor from a unit vector.

    X = C \sum_{d=0}^D (-1)^d \frac{(2n-2d-1)!!}{(2n-1)!!}
    \{ \hat{\bm a}^{\otimes^{n-2d}}\otimes \bm I^{\otimes d} \},

    where $D = n/2$ for even $n$ and $D = (n-1)/2$ for odd $n$.

    The constant $C$ is a normalization factor.

    Args:
        a: The unit vector.
        n: Rank of the natural tensor to create.
        normalize: Normalization type.
            If `unity`, $C = \frac{(2n-1)!!}{l!}$ is used for normalization.
            In this case, an n-contraction between the output natural tensor and an
            arbitrary unit vector `b` is equal to the Legendre polynomial of the angle
            between `a` and `b`. Namely: $out \odot^n b^{\otimes^n} = P_n(a \codt b)$.
            If `none`, no normalization is applied, i.e. $C = 1$.

    Returns:
            The rank-n natural tensor constructed from the unit vector.
    """
    # TODO, we can force to normalize `a` as a unit vector

    # For rank-0, return scalar 1. For rank-1, return the unit vector itself.
    if n == 0:
        return torch.tensor(1.0, dtype=a.dtype, device=a.device)
    elif n == 1:
        return a

    delta = dij(a.device)

    D = n // 2

    out = torch.zeros([3] * n, dtype=a.dtype, device=a.device)
    coeff = 1
    for d in range(D + 1):
        rule, symmetry, delta_indices = get_nt_from_vector_rule(n, d)

        # Get one tensor product
        prod = torch.einsum(rule, *([a] * (n - 2 * d)), *([delta] * d))

        # Symmetrize by summing over all unique permutations
        perms = get_permutations_delta(symmetry, delta_indices)
        prod = torch.sum(torch.stack([prod.permute(p) for p in perms]), dim=0)

        out = out + coeff * prod

        # Update the coefficient
        # coeff = (-1) ** d / double_factorial(2 * n - 1, 2 * n - 2 * d - 1 + 2)
        coeff = -coeff / (2 * n - 2 * d - 1)

    if normalize == "unity":
        out = double_factorial(2 * n - 1) / factorial(n) * out
    elif normalize == "none":
        pass
    else:
        raise ValueError(f"Unknown normalization method: {normalize}")

    return out


def get_nt_from_vector_rule(n: int, d: int) -> tuple[str, str, str]:
    """
    Get the rule for creating a rank-n natural tensor from a unit vector.

    Args:
        n: Rank of the natural tensor to create.
        d: Number of deltas.

    Returns:
        rule: The rule for creating a rank-n natural tensor from a unit vector.
        symmetry: The symmetry of the rule.
        delta_indices: The delta indices.
    """
    a_right = letter_index(n - 2 * d)
    a_left = ",".join(a_right)

    delta = double_index(d, start=n - 2 * d)
    delta_right = "".join(delta)
    delta_left = ",".join(delta) if delta else ""
    if a_left and delta_left:
        delta_left = "," + delta_left

    rule = f"{a_left}{delta_left}->{a_right}{delta_right}"
    symmetry = "x" * (n - 2 * d) + "".join(repeat_double_index(d))
    delta_indices = letter_index(d)

    return rule, symmetry, delta_indices
