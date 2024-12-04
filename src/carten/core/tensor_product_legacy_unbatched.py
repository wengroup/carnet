"""
Tensor product between two natural tensors, without considering batching dimensions.
"""

import itertools

import torch
from torch import Tensor

from carten.core.reduce import remove_trace
from carten.core.utils import dij, eijk, letter_index
from carten.natural_tensor import NaturalTensors


class TensorProduct:
    """
    Tensor product between two set of natural tensors.

    Resulting tensors of the same rank are grouped together.
    """

    def __init__(self, min_rank: int = None, max_rank: int = None):
        self.min_rank = min_rank
        self.max_rank = max_rank

    def __call__(self, nts1: NaturalTensors, nts2: NaturalTensors) -> NaturalTensors:
        """
        Compute the tensor product between two sets of natural tensors.
        """
        products = []
        for t1, t2 in itertools.product(nts1.chunks, nts2.chunks):
            p = tp(t1, t2, min_rank=self.min_rank, max_rank=self.max_rank)
            products.extend(p)

        return NaturalTensors.from_sequence(products)

    def __str__(self):
        return f"TensorProduct(min_rank={self.min_rank}, max_rank={self.max_rank})"


def tp_shaped_chunk(
    S: Tensor, T: Tensor, start_dim: int, min_rank: int = None, max_rank: int = None
) -> Tensor:
    """
    Compute the tensor product between two shaped chunked natural tensors.

    Each chunked tensor should be of shape (*leading_shape, mul, *ending_shape).
    `leading_shape` can be any shape, such as the batch dimensions. `mul` is the
    multiplicity. `ending_shape` is the shape of the natural tensor. For scalar, it
    should be (1,); for a vector, it should be (3,); for a general natural tensor, it
    should be (3,3,...,3) with the number of 3 equal to the rank of the natural tensor.
    `mul` and `*ending_shape` are flattened to a single dimension.

    Args:
        start_dim: the dimension to start the tensor product. This should be the
            dimension of the multiplicity `mul`.
    """


def tp(
    S: Tensor, T: Tensor, min_rank: int = None, max_rank: int = None
) -> list[Tensor]:
    """
    Tensor product of two natural tensors S and T.

    If S is a natural tensor with rank m and T is a natural tensor with rank n, then
    the tensor product S x T is a natural tensor with rank m + n, and its irreducible
    representations are a sequence of natural tensors with rank
    |m-n|, |m-n|+1, ..., 0, ..., m+n-1, m+n. The total number is 2 * min(m, n) + 1.

    Args:
        S: a natural tensor
        T: a natural tensor
        min_rank: minimum rank of the natural tensor from the tensor product to keep.
            Tensors with rank smaller than this are ignored. `min_rank` should be
            |m-n| <= min_rank <= m+n. If None, set to |m-n|.
        max_rank: maximum rank of the natural tensor from the tensor product to keep.
            Tensors with rank larger than this are ignored. `max_rank` should be
            |m-n| <= max_rank <= m+n. If None, set to m+n.

    Note:
        the tensor product is not commutative, i.e. S x T is not the same as T x S.

    Reference:
        A. Zee, Group Theory in a Nutshell for Physicists, Princeton University Press,
        2016. Page 208:
        Given two totally symmetric traceless natural tensors, S_{i1} ...{ij} and
        T_{k1} ..._{kj'}, one with j indices, the other with j' indices, the product is
        then a tensor with j + j' indices. We first symmetrize this and take out its
        trace. We get the irreducible representation labeled by j + j'. Next, contract
        with e_ikl , where i is an index on S and k an index on T. We trade two indices,
        i and k, for one index l, and hence end up with a tensor with j + j' - 1
        indices. We get the irreducible representation labeled by j + j' - 1. We repeat
        this process until there is nothing left to work with.

    Returns:
        A list of irreducible representations (natural tensors) of the tensor product
        between S and T. The irreducible representations are ordered by their ranks,
        from smallest to largest.

    """

    if min_rank is None:
        min_rank = abs(S.ndim - T.ndim)
    else:
        if min_rank < abs(S.ndim - T.ndim) or min_rank > S.ndim + T.ndim:
            raise ValueError(
                f"Expect `min_rank` to be between |S.ndim - T.ndim| and "
                f"S.ndim + T.ndim, but got {min_rank}"
            )
    if max_rank is None:
        max_rank = S.ndim + T.ndim
    else:
        if max_rank < abs(S.ndim - T.ndim) or max_rank > S.ndim + T.ndim:
            raise ValueError(
                f"Expect `max_rank` to be between |S.ndim - T.ndim| and "
                f"S.ndim+T.ndim, but got {max_rank}"
            )

    total_rank = S.ndim + T.ndim
    allowed_contraction = min(S.ndim, T.ndim)

    irreps = []
    for i in range(allowed_contraction + 1):
        num_delta = allowed_contraction - i

        # irrep of the antisymmetric part (after num_delta delta contractions)
        if (
            min_rank <= total_rank - 2 * num_delta - 1 <= max_rank
            and num_delta != allowed_contraction
        ):
            A = get_asym_part(S, T, num_delta)
            N2 = remove_trace(A)
            irreps.append(N2)

        # irrep of the symmetric part (after num_delta delta contractions)
        if min_rank <= total_rank - 2 * num_delta <= max_rank:
            U = get_sym_part(S, T, num_delta)
            N1 = remove_trace(U)
            irreps.append(N1)

    return irreps


def get_sym_part(S: Tensor, T: Tensor, num_delta: int = 0) -> Tensor:
    """
    Fully symmetrize the tensor product of two symmetrical tensors S and T.

    To fully symmetrize two symmetrical tensors S_ijk... and T_pqr..., we do it in a
    two-step process:
    - first, contract S and T `num_delta` times, and
    - second, symmetrize the resulting tensor.

    Args:
        S: a symmetrical tensor
        T: a symmetrical tensor
        num_delta: number of times to contract S and T with delta before symmetrizing.

    Returns:
        The fully symmetrized part of the tensor product of S and T.
    """
    if num_delta > min(S.ndim, T.ndim):
        raise ValueError(
            "Expect `num_delta` to be <= min(S.ndim, T.ndim), but got " f"{num_delta}"
        )

    S_indices = letter_index(S.ndim)
    T_indices = letter_index(T.ndim, start=S.ndim)

    left = get_delta_contraction_rule(S_indices, T_indices, num_delta)

    # indices after contraction
    S_indices = S_indices[num_delta:]
    T_indices = T_indices[num_delta:]

    right_indices = get_sym_rules_2(S_indices, T_indices)
    rules = [f"{left}->{right}" for right in right_indices]

    data = [S, T]
    if num_delta > 0:
        delta = dij()
        data = [delta for _ in range(num_delta)] + data

    # TODO, it might be faster to separate it into two steps: 1. contract S and T to
    #  get U, and 2. symmetrize U. This way, we can use U for all symmetrization rules.
    # Yes, definitely do it
    symmetrized = torch.mean(
        torch.stack([torch.einsum(r, *data) for r in rules]), dim=0
    )

    return symmetrized


def get_asym_part(S: Tensor, T: Tensor, num_delta: int = 0) -> Tensor:
    """
    Fully symmetrize the asymmetrical part of the tensor product of two natural tensors
    S and T.

    This is achieved in two steps:
    - first get the asymmetrical part of the tensor product of S and T (possibly with
      some delta contractions);
    - second, symmetrize the resulting tensor.

    For example, given two natural tensors S_abc and T_def, the asymmetrical part
    is obtained as (when num_delta=0):
        U_zbcef = e_zad S_abc T_def

    Given that S and T are traceless, it is only necessary to choose one index from S
    and another index from T. This will be the only asymmetrical part of the tensor
    product between S and T. There is no need to choose two indices from S (or T); the
    resulting tensor will be 0, because T (or S) is traceless.

    However, the resulting tensor U is not a fully symmetrical tensor. It will be
    symmetrical in e and f (indices originally coming from S), and also symmetrical
    in g and i (indices originally coming from T). But it is not symmetrical between c
    and the other indices. As a result, in the second step, we need to symmetrize w.r.t.
    the asymmetrical indices, not all the indices.

    Args:
        S: a natural tensor
        T: a natural tensor
        num_delta: number of times to contract S and T with delta before multiplying
            the antisymmetric tensor epsilon.

    Returns:
        The fully symmetrized part of the antisymmetrical part of the tensor product of
        S and T. This should be a natural tensor.
    """
    if num_delta > 1 + min(S.ndim, T.ndim):
        raise ValueError(
            f"Expect `num_delta` to be <= 1 + min(S.ndim, T.ndim), but got {num_delta}"
        )

    S_indices = letter_index(S.ndim)
    T_indices = letter_index(T.ndim, start=S.ndim)

    left = get_epsilon_delta_contraction_rule(S_indices, T_indices, num_delta)

    # indices after delta contraction
    S_indices = S_indices[1 + num_delta :]
    T_indices = T_indices[1 + num_delta :]

    right_indices = get_sym_rules_3("z", S_indices, T_indices)
    rules = [f"{left}->{right}" for right in right_indices]

    if num_delta > 0:
        d = dij()
        delta = [d for _ in range(num_delta)]
    else:
        delta = []
    data = [eijk()] + delta + [S, T]

    # TODO, it might be faster to separate it into two steps: 1. contract S and T to
    #  get U, and 2. symmetrize U. This way, we can use U for all symmetrization rules.
    symmetrized = torch.mean(
        torch.stack([torch.einsum(r, *data) for r in rules]), dim=0
    )

    return symmetrized


def get_delta_contraction_rule(group_a: str, group_b: str, num_delta: int = 0) -> str:
    """
    Get rules to contract two groups of indices with (multiple) delta.

    Example:
        >>> get_delta_contraction_rule("ijk", "pqr", num_delta=0)
        "ijk,pqr"
        >>> get_delta_contraction_rule("ijk", "pqr", num_delta=1)
        "ip,ijk,pqr"
        >>> get_delta_contraction_rule("ijk", "pqr", num_delta=2)
        "ip,jq,ijk,pqr"

    Args:
        group_a: a string of indices, e.g. "ijk"
        group_b: a string of indices, e.g. "pqr"
        num_delta: number of times to contract with delta

    Returns:
        A string of indices, e.g. "ip,jq,ijk,pqr", where the last two set of indices are
        the original indices, and the two sets of indices before them are the indices
        to be contracted with the delta tensors.

    """
    if num_delta == 0:
        return f"{group_a},{group_b}"
    else:
        delta_indices = ",".join([group_a[i] + group_b[i] for i in range(num_delta)])
        return f"{delta_indices},{group_a},{group_b}"


def get_epsilon_delta_contraction_rule(
    group_a: str, group_b: str, num_delta: int = 0
) -> str:
    """
    Get rules to contract two groups of indices with one epsilon and (multiple) delta.

    Example:
        >>> get_epsilon_delta_contraction_rule("ijk", "pqr", num_delta=0)
        "zip,ijk,pqr"
        >>> get_epsilon_delta_contraction_rule("ijk", "pqr", num_delta=1)
        "zjq,ip,ijk,pqr"
        >>> get_epsilon_delta_contraction_rule("ijk", "pqr", num_delta=2)
        "zkr,ip,jq,ijk,pqr"

    Warning:
        This assumes that letter `z` is not used in the indices.

    Args:
        group_a: a string of indices, e.g. "ijk"
        group_b: a string of indices, e.g. "pqr"
        num_delta: number of times to contract with delta

    Returns:
        A string of indices, e.g. "zkr,ip,jq,ijk,pqr", where the last two set of indices
        are the original indices; the first set of indices are the indices associated
        with the epsilon tensor; and the two sets of indices in the middle are the
        indices to be contracted with the delta tensors.
    """
    indices = get_delta_contraction_rule(group_a, group_b, num_delta)
    epsilon_indices = "z" + group_a[num_delta] + group_b[num_delta]
    return f"{epsilon_indices},{indices}"


def get_sym_rules_2(group_a: str, group_b: str) -> list[str]:
    """Get all possible symmetrizing rules between two groups of symmetric indices.

    The indices within each group are symmetric, but not between the two groups. As a
    result, we just need to symmetrize between groups.

    Algorithm:
        Suppose S has m indices (ijk..) and T has n indices (pqr...).
        1. Given m+n positions to place the indices, choose m positions for S.
        2. Place the m indices of S (ijk...) in the chosen positions. The order of the
           indices does not matter, because S is fully symmetric.
        3. For the remaining n positions, place indices of T (pqr...). Again, the order
           does not matter.
        4. Average over all possible placements of indices for S and T.

    Example:
        Given two symmetrical tensors S_ij and T_pq, we have (for each line, first
        equality switches pq, second equality switches ij, and third equality switches
        both):
        - S_ij T_pq = S_ij T_qp = S_ji T_pq = S_ji T_qp
        - S_ip T_jq = S_iq T_jp = S_jp T_iq = S_jq T_ip
        - S_ip T_qj = S_iq T_pj = S_jp T_qi = S_jq T_pi
        - S_pi T_jq = S_qi T_jp = S_pj T_iq = S_qj T_ip
        - S_pi T_qj = S_qi T_pj = S_pj T_qi = S_qj T_pi
        - S_pq T_ij = S_qp T_ij = S_pq T_ji = S_qp T_ji

        Then the symmetrized tensor is:
            (S_ij T_pq + S_ip T_jq + S_ip T_qj + S_pi T_jp + S_pi T_qj + S_pq T_ij) / 6

    Args:
        group_a: a string of indices, e.g. "ijk"
        group_b: a string of indices, e.g. "pqr"

    Returns:
        A list of all symmetrizing rules, e.g. ["ijkpqr", "ijpkqr", ...]
    """

    possible_pos = set(range(len(group_a) + (len(group_b))))

    rules = []
    for a_pos in itertools.combinations(possible_pos, len(group_a)):
        b_pos = possible_pos - set(a_pos)

        arrange = ["_"] * len(possible_pos)
        for i, p in enumerate(a_pos):
            arrange[p] = group_a[i]
        for i, p in enumerate(b_pos):
            arrange[p] = group_b[i]

        rules.append("".join(arrange))

    return rules


def get_sym_rules_3(group_a: str, group_b: str, group_c: str) -> list[str]:
    """
    Get all possible symmetrizing rules between three groups of symmetric indices.


    Algorithm:
        Similar to `get_sym_rules_2`, but with three groups of indices.

    Args:
        group_a: a string of indices, e.g. "ij"
        group_b: a string of indices, e.g. "pq"
        group_c: a string of indices, e.g. "xy"

    Returns:
        A list of all symmetrizing rules, e.g. ["ijpxyq", "ipjqxy", ...]
    """

    possible_pos = set(range(len(group_a) + len(group_b) + len(group_c)))

    rules = []
    for a_pos in itertools.combinations(possible_pos, len(group_a)):
        remaining_pos = possible_pos - set(a_pos)

        for b_pos in itertools.combinations(remaining_pos, len(group_b)):
            c_pos = remaining_pos - set(b_pos)

            arrange = ["_"] * len(possible_pos)
            for i, p in enumerate(a_pos):
                arrange[p] = group_a[i]
            for i, p in enumerate(b_pos):
                arrange[p] = group_b[i]
            for i, p in enumerate(c_pos):
                arrange[p] = group_c[i]

            rules.append("".join(arrange))

    return rules
