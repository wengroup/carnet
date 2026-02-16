"""Natural tensors of unit vector.

This is a faster implementation compared to unit_vector_1.py.
Here, N^l = H \cdot a^l, where H can be precomputed, and this can avoid loop.
"""

from fractions import Fraction

import torch
from natt.EGH import create_delta_epsilon_tensors
from natt.evaluate import evaluate_tensors
from natt.ops import simplify_linear_combination
from natt.symbolic import LinearCombination
from natt.symmetrize import get_permutations_delta
from natt.utils import double_factorial, factorial, letter_index
from torch import Tensor

from carnet.core.unit_vector_1 import get_nt_from_vector_rule

H_AND_RULE_CACHE = {}


def get_polyadics_from_vector(a: Tensor, L: int, normalize: str = "unity"):
    r"""
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
    return torch.cat(
        [get_nt_from_vector(a, l, normalize, flatten=True) for l in range(L + 1)],
        dim=-1,
    )


def get_nt_from_vector(
    a: Tensor, l: int, normalize: str = "unity", flatten: bool = False
) -> Tensor:
    r"""
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

    batch_dims = a.shape[:-1]

    # For rank-0, return scalar 1. For rank-1, return the unit vector itself.
    if l == 0:
        out = torch.ones(batch_dims, dtype=a.dtype, device=a.device)
        return out.view(batch_dims + (1,))
    elif l == 1:
        return a

    # Get H, and rule
    global H_AND_RULE_CACHE
    if (l, normalize) not in H_AND_RULE_CACHE:
        H, rule = get_H_numerical(l, normalize)
        H = H.to(device=a.device)
        H_AND_RULE_CACHE[(l, normalize)] = (H, rule)
    else:
        H, rule = H_AND_RULE_CACHE[(l, normalize)]

    # Compute
    out = torch.einsum(rule, [H] + [a] * l)

    if flatten:
        return out.view(batch_dims + (-1,))
    else:
        return out


def get_H_numerical(l: int, normalize: str = "unity") -> tuple[Tensor, str]:
    """Numerical H.

    Args:
        l: The rank of the output tensor.
        normalize: The normalization method.
            If `unity`, the output is normalized such that the l3 fold contraction of
            the output tensor with a unit vector yields 1.
            If `none`, no normalization is applied.
    """
    H, X_idx, Z_idx = get_H(l)

    H = simplify_linear_combination(H)

    # The indices of H_numerical will  be {Z_idx}{X_idx}
    H_numerical = evaluate_tensors(H, mode="H")

    if normalize == "unity":
        H_numerical *= double_factorial(2 * l - 1) / factorial(l)
    elif normalize == "none":
        pass
    else:
        supported = ["none", "unity"]
        raise ValueError(
            f"Unknown normalization method: {normalize}. Supported are: {supported}."
        )

    X_separate = ",...".join(list(X_idx))
    rule = f"{Z_idx}{X_idx},...{X_separate}->...{Z_idx}"

    return H_numerical, rule


def get_H(l: int) -> tuple[LinearCombination, str, str]:
    """Symbolic H."""

    out = []
    for d in range(l // 2 + 1):

        coeff = Fraction(
            (-1) ** d, double_factorial(2 * l - 1, 2 * l - 2 * d - 1 + 2).item()
        )

        all_rules = get_H_rules(l, d)

        # create tensor products of deltas for each rule
        tensors = [
            create_delta_epsilon_tensors(ru["ra"] + ru["aa"] + ru["rr"], factor=coeff)
            for ru in all_rules
        ]

        # extend them to sum up later
        out.extend(tensors)

    H = LinearCombination(*out)

    # Note, this should exactly the same as those in `get_H_rules()`
    X_idx = letter_index(l, upper_case=True)
    Z_idx = letter_index(l)

    return H, X_idx, Z_idx


def get_H_rules(l: int, d: int) -> list[dict[str, list[str]]]:
    """
    Get the rules to construct { delta_{ra}^{l-2d} \delta_{aa}^d } delta_rr^d

    Refer to: H_tp.py for more information. This is analogous to that, but more simple.

    Args:
         l: rank of the output tensor.
         d:

    Returns:
        Each dict gives the indices for the Kronecker delta tensors d_ra, d_sa, d_aa,
        and d_rs, which can be used to create the left-hand-side of the einsum rule.
        The r_indices, s_indices, and a_indices can b
    """

    # r indices for delta_ra
    r_idx = letter_index(l, upper_case=True)
    a_idx = letter_index(l)

    r_ra_idx = r_idx[: l - 2 * d]
    r_rr_idx = r_idx[l - 2 * d :]
    rr_pairs = [r_rr_idx[2 * i] + r_rr_idx[2 * i + 1] for i in range(d)]

    _, symmetry, delta_indices = get_nt_from_vector_rule(l, d)
    all_perms = get_permutations_delta(symmetry, delta_indices)

    all_rules = []
    for perm in all_perms:
        # Permute the a indices to symmetrize the output, namely considering the
        # curly braces {}. No need to permute the r indices

        #
        # The below is the same as
        # indices = [letters[perm.index(i)] for i in range(n)]
        p_a = [x for _, x in sorted(zip(perm, a_idx))]

        # a index in delta_ra and in delta_aa
        a_ra_idx = p_a[: l - 2 * d]
        a_aa_idx = p_a[l - 2 * d :]

        # Pairs of indices for delta tensors
        ra_pairs = [r + a for r, a in zip(r_ra_idx, a_ra_idx)]
        aa_pairs = [a_aa_idx[2 * i] + a_aa_idx[2 * i + 1] for i in range(d)]

        all_rules.append({"ra": ra_pairs, "aa": aa_pairs, "rr": rr_pairs})

    return all_rules


if __name__ == "__main__":
    torch.manual_seed(35)

    a = torch.randn(2, 3)
    out = get_nt_from_vector(a, l=2)

    # Check symbolic H
    l = 3
    H, X_idx, Z_idx = get_H(l)
    H = simplify_linear_combination(H)
    print(f"H (l={l}):", H)
