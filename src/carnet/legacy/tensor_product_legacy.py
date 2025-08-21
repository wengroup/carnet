"""Tensor product between two natural tensors, with batching support.

This is using the symmetrizing and then removing the trace method.
"""

import itertools

import torch
from torch import Tensor

from carnet.core.utils import dij, eijk, letter_index
from carnet.legacy.natural_tensor import NaturalTensors
from carnet.legacy.reduce import remove_trace


class TensorProduct:
    """
    Tensor product between two set of natural tensors.

    Output tensors of the same rank are grouped together.


    Args:
        out_ranks: ranks of the output tensors to keep. If None, the output tensor of
            all ranks are kept.
    """

    def __init__(self, out_ranks: list[int] = None):
        self.out_ranks = out_ranks

    def __call__(self, nts1: NaturalTensors, nts2: NaturalTensors) -> NaturalTensors:
        """
        Compute the tensor product between two sets of natural tensors.
        """
        products = []

        for t1, r1 in zip(nts1.shaped_chunks, nts1.chunk_ranks):
            for t2, r2 in zip(nts2.shaped_chunks, nts2.chunk_ranks):
                p = tp(
                    t1,
                    t2,
                    rank_S=r1,
                    rank_T=r2,
                    out_ranks=self.out_ranks,
                )
                products.extend(p)

        return NaturalTensors.from_sequence(products, start_dim=2)

    def __str__(self):
        return f"TensorProduct(out_ranks={self.out_ranks})"


def tp(
    S: Tensor,
    T: Tensor,
    rank_S: int,
    rank_T: int,
    out_ranks: list[int] = None,
) -> list[Tensor]:
    """
    Tensor product of chunked tensors T and S.


    If S is a natural tensor with rank p and T is a natural tensor with rank q, then
    the tensor product S x T is a natural tensor with rank p + q, and its irreducible
    representations are a sequence of natural tensors with rank
    |p-q|, |p-q|+1, ..., 0, ..., p+q-1, p+q. The total number is 2 * min(p, q) + 1.

    Args:
        S: a tensor of shape (..., m_S, 3,...,3), where m_S is the multiplicity of the
            tensor and the number of 3's is equal to the rank of the tensor.
        T: a tensor of shape (..., m_T, 3,...,3), where m_T is the multiplicity of the
            tensor and the number of 3's is equal to the rank of the tensor.
        rank_S: the rank of S.
        rank_T: the rank of T.
        out_ranks: ranks of the output tensors to keep. If None, the output tensor of
            all ranks are kept.

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
        between S and T.


        The irreducible representations are ordered by their ranks,
        from smallest to largest.
    """
    min_rank = abs(rank_S - rank_T)
    max_rank = rank_S + rank_T

    allowed_ranks = list(range(min_rank, max_rank + 1))
    if out_ranks is None:
        out_ranks = allowed_ranks
    else:
        for r in out_ranks:
            if r not in allowed_ranks:
                raise ValueError(
                    f"rank {r} for output tensor is not allowed. "
                    f"Allowed ranks are {allowed_ranks}."
                )

    total_rank = rank_S + rank_T
    allowed_contraction = min(rank_S, rank_T)

    irreps = []
    for i in range(allowed_contraction + 1):
        num_delta = allowed_contraction - i

        # irrep of the antisymmetric part (after num_delta delta contractions)
        if (
            total_rank - 2 * num_delta - 1 in out_ranks
            and num_delta != allowed_contraction
        ):
            A = get_asym_part(S, T, rank_S, rank_T, num_delta)

            start_dim = A.ndim - (rank_S + rank_T - num_delta * 2 - 1)
            N2 = remove_trace(A, start_dim)

            irreps.append(N2)

        # irrep of the symmetric part (after num_delta delta contractions)
        if total_rank - 2 * num_delta in out_ranks:
            U = get_sym_part(S, T, rank_S, rank_T, num_delta)

            start_dim = U.ndim - (rank_S + rank_T - num_delta * 2)
            N1 = remove_trace(U, start_dim)

            irreps.append(N1)

    return irreps


def get_sym_part(
    S: Tensor,
    T: Tensor,
    rank_S: int,
    rank_T: int,
    num_delta: int = 0,
) -> Tensor:
    """
    Fully symmetrize the tensor product of two symmetrical tensors S and T.

    The batching dimensions of S and T should be the same. They are intact and carried
    over to the result.

    # TODO we can enable other rules for the multiplicity dimensions.
    The multiplicity dimensions of S and T are treated separately. For example, if S
    has multiplicity 2 and T has multiplicity 3, then the result will have multiplicity
    6.

    To fully symmetrize two symmetrical tensors S_ijk... and T_pqr..., we do it in a
    two-step process:
    - first, contract S and T `num_delta` times, and
    - second, symmetrize the resulting tensor.

    Args:
        S: a symmetrical tensor of shape (..., m_S, 3,...,3), where m_S is the
            multiplicity of the tensor and the number of 3's is equal to the rank of
            the tensor.
        T: a symmetrical tensor of shape (..., m_T, 3,...,3), where m_T is the
            multiplicity of the tensor and the number of 3's is equal to the rank of
            the tensor.
        rank_S: the rank of S.
        rank_T: the rank of T.
        num_delta: number of times to contract S and T with delta before symmetrizing.

    Returns:
        The fully symmetrized part of the tensor product of S and T.
    """
    if num_delta > min(rank_S, rank_T):
        raise ValueError(
            "Expect `num_delta` to be <= min(rank_S, rank_T), but got " f"{num_delta}"
        )

    mul_S = S.shape[-rank_S - 1]
    mul_T = T.shape[-rank_T - 1]

    batch_shape_S = S.shape[: -rank_S - 1]
    batch_shape_T = T.shape[: -rank_T - 1]
    if batch_shape_S != batch_shape_T:
        raise ValueError(
            f"Expect the batching dimensions of S and T to be the same, but got "
            f"{batch_shape_S} and {batch_shape_T}"
        )

    S_indices = letter_index(rank_S + 1)  # +1 for the multiplicity dimension
    T_indices = letter_index(rank_T + 1, start=rank_S + 1)

    left = get_delta_contraction_rule(S_indices, T_indices, num_delta)

    # +1 to consider the multiplicity dimension
    S_indices_after = S_indices[num_delta + 1 :]
    T_indices_after = T_indices[num_delta + 1 :]

    right_indices = get_sym_rules_2(S_indices_after, T_indices_after)

    # S_indices[0] and T_indices[0] are the multiplicity dimensions
    rules = [
        f"{left}->...{S_indices[0]}{T_indices[0]}{right}" for right in right_indices
    ]

    data = [S, T]
    if num_delta > 0:
        delta = dij()
        data = [delta for _ in range(num_delta)] + data

    # TODO, it might be faster to separate it into two steps: 1. contract S and T to
    #  get U, and 2. symmetrize U. This way, we can use U for all symmetrization rules.
    #
    # TODO, we should do exactly the above to avoid many einsum calls. We just get U,
    #  and then torch.permute the indices according to the symmetrization rules.

    # TODO, NEED to figure out how to average, not over the batch / multi direction
    symmetrized = torch.mean(
        torch.stack([torch.einsum(r, *data) for r in rules]), dim=0
    )

    # combine the two multiplicity dimensions as one
    tensor_shape = [3] * (rank_S + rank_T - 2 * num_delta)
    symmetrized = symmetrized.reshape(*batch_shape_S, mul_S * mul_T, *tensor_shape)

    return symmetrized


def get_asym_part(
    S: Tensor, T: Tensor, rank_S: int, rank_T: int, num_delta: int = 0
) -> Tensor:
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
        S: a natural tensor of shape (..., m_S, 3,...,3), where m_S is the multiplicity
            of the tensor and the number of 3's is equal to the rank of the tensor.
        T: a natural tensor of shape (..., m_T, 3,...,3), where m_T is the multiplicity
            of the tensor and the number of 3's is equal to the rank of the tensor.
        rank_S: the rank of S.
        rank_T: the rank of T.
        num_delta: number of times to contract S and T with delta before multiplying
            the antisymmetric tensor epsilon.

    Returns:
        The fully symmetrized part of the antisymmetrical part of the tensor product of
        S and T. This should be a natural tensor.
    """
    if num_delta > 1 + min(rank_S, rank_T):
        raise ValueError(
            f"Expect `num_delta` to be <= 1 + min(rank_S, rank_T), but got {num_delta}"
        )

    mul_S = S.shape[-rank_S - 1]
    mul_T = T.shape[-rank_T - 1]

    batch_shape_S = S.shape[: -rank_S - 1]
    batch_shape_T = T.shape[: -rank_T - 1]
    if batch_shape_S != batch_shape_T:
        raise ValueError(
            f"Expect the batching dimensions of S and T to be the same, but got "
            f"{batch_shape_S} and {batch_shape_T}"
        )

    S_indices = letter_index(rank_S + 1)
    T_indices = letter_index(rank_T + 1, start=rank_S + 1)

    left = get_epsilon_delta_contraction_rule(S_indices, T_indices, num_delta)

    # +2: 1 for epsilon and 1 for multiplicity dimension
    S_indices_after = S_indices[num_delta + 2 :]
    T_indices_after = T_indices[num_delta + 2 :]

    right_indices = get_sym_rules_3("z", S_indices_after, T_indices_after)

    rules = [
        f"{left}->...{S_indices[0]}{T_indices[0]}{right}" for right in right_indices
    ]

    if num_delta > 0:
        d = dij()
        delta = [d for _ in range(num_delta)]
    else:
        delta = []
    data = [eijk()] + delta + [S, T]

    # TODO, it might be faster to separate it into two steps: 1. contract S and T to
    # get U, and 2. symmetrize U. This way, we can use U for all symmetrization rules.
    symmetrized = torch.mean(
        torch.stack([torch.einsum(r, *data) for r in rules]), dim=0
    )

    # combine the two multiplicity dimensions as one
    tensor_shape = [3] * (rank_S + rank_T - 2 * num_delta - 1)
    symmetrized = symmetrized.reshape(*batch_shape_S, mul_S * mul_T, *tensor_shape)

    return symmetrized


def get_delta_contraction_rule(group_a: str, group_b: str, num_delta: int = 0) -> str:
    """
    Get rules to contract two groups of indices with (multiple) delta.

    Example:
        >>> get_delta_contraction_rule("ijk", "pqr", num_delta=0)
        "...ijk, ...pqr"
        >>> get_delta_contraction_rule("ijk", "pqr", num_delta=1)
        "jq, ...ijk, ...pqr"
        >>> get_delta_contraction_rule("ijk", "pqr", num_delta=2)
        "jq, kr, ...ijk, ...pqr"

    The `...` represents whatever dimensions (e.g. batching) to be carried over.

    Warning:
        This assumes that letters `y` and `z` are not used in the indices. No check is
        performed.

    Args:
        group_a: a string of indices, e.g. "ijk". The first index represents the
            multiplicity dimension (which will not be contracted), and the rest are
            the indices for contraction.
        group_b: a string of indices, e.g. "pqr". The first index represents the
            multiplicity dimension (which will not be contracted), and the rest are
            the indices for contraction.
        num_delta: number of times to contract with delta.

    Returns:
        A string of indices, e.g. "jq, kr, ...ijk, ...pqr", where the last two set of
        indices are the original indices, and the indices before them are the indices
        to be contracted with the delta tensors.
    """
    if num_delta == 0:
        return f"...{group_a}, ...{group_b}"
    else:
        delta_indices = ", ".join(
            [group_a[i] + group_b[i] for i in range(1, 1 + num_delta)]
        )
        return f"{delta_indices}, ...{group_a}, ...{group_b}"


def get_epsilon_delta_contraction_rule(
    group_a: str, group_b: str, num_delta: int = 0
) -> str:
    """
    Get rules to contract two groups of indices with one epsilon and (multiple) delta.

    Example:
        >>> get_epsilon_delta_contraction_rule("ijk", "pqr", num_delta=0)
        "zjq, ...ijk, ...pqr"
        >>> get_epsilon_delta_contraction_rule("ijk", "pqr", num_delta=1)
        "zkr, jq, ...ijk, ...pqr"

    Warning:
        This assumes that letter `z` is not used in the indices.

    Args:
        group_a: a string of indices, e.g. "ijk". The first index represents the
            multiplicity dimension (which will not be contracted), and the rest are
            the indices for contraction.
        group_b: a string of indices, e.g. "pqr". The first index represents the
            multiplicity dimension (which will not be contracted), and the rest are
            the indices for contraction.
        num_delta: number of times to contract with delta.

    Returns:
        A string of indices, where the last two set of indices are the original indices;
        the first set of indices are the indices to be contracted with the epsilon
        tensor; and the rest indices in the middle are the
        indices to be contracted with the delta tensors.
    """
    indices = get_delta_contraction_rule(group_a, group_b, num_delta)
    epsilon_indices = "z" + group_a[num_delta + 1] + group_b[num_delta + 1]
    return f"{epsilon_indices}, {indices}"


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

    For example Given two symmetrical tensors S_ij and T_pq, we have (for each line,
    first equality switches pq, second equality switches ij, and third equality switches
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

    Example:
        >>> get_sym_rules_2('ij', 'pq')
        ['ijpq', 'ipjq', 'ipqj', 'pijq', 'piqj', 'pqij']

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

    Example:
        >>> get_sym_rules_3('z', 'pq', 'xy')
        ['zpqxy', 'zpxqy', 'zpxyq', 'zxpqy', zxpyq', 'zxypq',
         'pzqxy', 'pzxqy', 'pzxyq', 'xzpqy', 'xzpyq', 'xzypq',
         ...
         'pqxyz', 'pzxqyz', 'pxyqz', 'xpqyz', 'xpyqz', 'xypqz',
        ]

    Returns:
        A list of all symmetrizing rules, e.g. ["zpqxy", "zpxqy", ...]
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
