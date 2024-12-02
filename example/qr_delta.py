"""
Same as qr_delta_double_epsilon-2.py, but using no epsilon contraction.
"""

import torch
from example.utils import tp_with_delta_epsilon

from carten.reduce import get_permutations_2, symmetrize_and_remove_trace
from carten.utils import (
    is_symmetric_traceless,
    find_independent_tensors,
    letter_index,
)


def get_rules(rank: int, num_delta: int) -> list[tuple[str, str]]:
    """
    Rules to contract a tensor with multiple delta tensors and a double contraction with
    epsilon tensor.

    Args:
        rank: the rank of the tensor
        num_delta: the number of delta

    Returns:
        rules, e.g. ['xab,cd,ef,abcdefg->xg', xab,cd,ef,xg->abcdefg'] with deltas,
        epsilon, and the tensor indices in order. The first being the forward
        contraction, and the second being the backward contraction embed the tensor.
    """

    perms = get_permutations_2(rank, num_delta)

    t_indices = letter_index(rank)

    # TODO, this depends on the order of the indices get_permutations_2 returns, where
    #   we put the remaining indices of t at the front, and the contracted indices at
    start_idx = rank - 2 * num_delta

    rules = []
    for p in perms:
        delta = [
            t_indices[p.index(start_idx + 2 * i)]
            + t_indices[p.index(start_idx + 2 * i + 1)]
            for i in range(num_delta)
        ]
        remaining = sorted(set(t_indices) - set("".join(delta)))

        r1 = f"{','.join(delta)},{t_indices}->{''.join(remaining)}"
        r2 = f"{','.join(delta)},{''.join(remaining)}->{t_indices}"
        rules.append((r1, r2))

    return rules


if __name__ == "__main__":
    ### settings
    torch.manual_seed(0)
    rank = 4
    num_delta = 1
    num_epsilon = 0  # should never change this
    ###

    t = torch.rand([3] * rank)

    # get the rules
    rules = get_rules(t.ndim, num_delta)
    for i, r in enumerate(rules):
        print("Rule", i, r)

    # get the natural tensors, and checking
    forward_rules = [r[0] for r in rules]
    candidates = [
        tp_with_delta_epsilon(t, num_delta, num_epsilon, r) for r in forward_rules
    ]
    natural_tensors = [symmetrize_and_remove_trace(c) for c in candidates]
    for t in natural_tensors:
        assert is_symmetric_traceless(t)

    # NOTE, we just choose the first as S_apq
    nt = natural_tensors[0]

    # embed the tensor back
    backward_rules = [r[1] for r in rules]
    embedded_tensors = [
        tp_with_delta_epsilon(nt, num_delta, num_epsilon, r) for r in backward_rules
    ]

    # find independent tensors
    _, independent_indices = find_independent_tensors(embedded_tensors, tolerance=1e-3)

    print(f"n={rank}, J={rank-2*num_delta}, num_delta={num_delta}")

    # TODO, the below is incorrect, we need to find g_pq and such
    ################################################################################
    # For a tensor T(n), we first extract the natural tensors, to get X^{p,j}, and then
    # embed the natural tensors back into a tensor space of rank n, get S^p(n).
    # We check that \sum_p S^p(n) = T(n).
    ################################################################################
    # select X^p_j of different p
    independent_X = [natural_tensors[i] for i in independent_indices]
    embedding_rules = [backward_rules[i] for i in independent_indices]
    S = [
        1 / 3 * tp_with_delta_epsilon(x, num_delta, num_epsilon, r)
        for x, r in zip(independent_X, embedding_rules)
    ]

    sum_S = torch.stack(S).sum(dim=0)

    print("sum t", t.sum())
    print("sum S", sum_S.sum())

    assert torch.allclose(t, sum_S, atol=1e-5), "The sum of S^p(n) is not equal to T(n)"

    ################################################################################
