"""
We use QR decomposition to find linearly independent natural tensors that are obtained
from higher dimensional space.

TODO, here we focus on multiple delta and a double contraction with a single epsilon:
e_aij d_kl d_mn T_ijklmnpq

We first perform contraction to and and symmetrize the remaining indices: apq,
then embed the tensor back to its original space X_ijklmnpq.

If we do not consider symmetrizing and removing trace, this is:
e_ai'j' e_aij d_kl d_k'l' d_mn dm'n' T_ijklmnpq

The e_ai'j' e_aij operation embeds the tensor back to its original space, i.e. exchanging
the index `a` with two new indices `i'` and `j'`.

In other words, after contraction, we add new indices (indicated by ') to the original
space.

in einsum notation, this is:
d # delta
e # epsilon

out = einsum('ai'j', aij, kl, k'l', mn, m'n', i'j'k'l'm'n'pq -> ijklmnpq', e, e, d, d, T)

But we need to consider the symmetrization and traceless condition. With that in mind,
we do it in the below way:
1. Contract with deltas and epsilons:
    S_apq = e_aij d_kl d_mn T_ijklmnpq,
    which can be done as:
        rule1 = "aij, kl, mn, ijklmnpq -> apq"
        einsum(rule1, e, d, d, T)
    In the meanwhile, we need to construct the for step 3 below:
        rule2 = "aij, kl, mn, apq -> ijklmnpq"
        einsum(rule2, e, d, d, S)
    As seen, the rules hare highly related, the ones associated with epsilon and delta
    are the same, and only need to switch the positions of the two associated with
    T_ijklmnpq and S_apq.
2. Symmetrize and remove the trace of in S_apq
3. Embed the tensor back to its original space
   X_i'j'k'l'm'n'pq = e_ai'j' d_k'l' dm'n' S_apq,
   which can be done use rule 2 above.


For backward embedding, we do not perform epsilon contraction, and the rule is:
1. Contract with deltas and epsilons:
    S_apq = e_aij d_kl d_mn dm'n' T_ijklmnpq,
    which can be done as:
        rule1 = "aij, kl, mn, ijklmnpq -> apq"
        einsum(rule1, e, d, d, T)
    In the meanwhile, we need to construct the for step 3 below:
        rule2 = "kl, mn, apq -> aklmnpq"
        einsum(rule2, d, d, S)
    As seen, the rules hare highly related, the ones associated with epsilon and delta
    are the same, and only need to switch the positions of the two associated with
    T_ijklmnpq and S_apq.
2. Symmetrize and remove the trace of in S_apq
3. Embed the tensor back to its original space
   X_ak'l'm'n'pq = d_k'l' dm'n' S_apq,
   which can be done use rule 2 above.
"""

# TODO, In qr_delta_double_epsilon.py, the above three steps are performed.
#  In that case, S_apq will be different for each choice of the delta and epsilon
#  indices. It turns this is not the correct way to check linear independence.
#  The correct way should be: using the same S_apq for all choices of delta and epsilon.
#  So, here the procedure is:
#  1. Create a symmetric traceless tensor S_apq
#  2. Embed the tensor back to its original space X_ijklmnpq, using rule2 above. (This
#     is step 3 above.)


import itertools

import torch
from example.utils2 import tp_with_delta_epsilon

from carten.reduce import get_permutations_2, symmetrize_and_remove_trace
from carten.utils import (
    check_symmetric_traceless,
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

        # choose two to contract with epsilon
        for comb in itertools.combinations(remaining, 2):
            epsilon_contracted = "".join(comb)
            uncontracted = "".join(sorted(set(remaining) - set(epsilon_contracted)))

            sep = "," if num_delta > 0 else ""
            r1 = f"x{epsilon_contracted}{sep}{','.join(delta)},{t_indices}->x{uncontracted}"

            # embed epsilon back
            r2 = f"x{epsilon_contracted}{sep}{','.join(delta)},x{uncontracted}->{t_indices}"

            # # TODO, For the backward embedding, we don't perform epsilon contraction
            # right = "x" + t_indices.replace(epsilon_contracted[0], "").replace(
            #     epsilon_contracted[1], ""
            # )
            # r2 = f"{','.join(delta)}{sep}x{uncontracted}->{right}"

            rules.append((r1, r2))

    return rules


if __name__ == "__main__":
    ### settings
    torch.manual_seed(0)
    rank = 7
    num_delta = 2
    num_epsilon = 1  # should never change this
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
        check_symmetric_traceless(t)

    # NOTE, we just choose the first as S_apq
    nt = natural_tensors[0]

    # embed the tensor back
    backward_rules = [r[1] for r in rules]
    embedded_tensors = [
        tp_with_delta_epsilon(nt, num_delta, num_epsilon, r) for r in backward_rules
    ]

    # find independent tensors
    _, independent_indices = find_independent_tensors(embedded_tensors)
    print("Num independent tensors:", len(independent_indices))

    print(f"n={rank}, J={rank-2*num_delta-1}, num_delta={num_delta}")
