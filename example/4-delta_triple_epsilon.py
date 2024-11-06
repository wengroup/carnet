"""
Double contraction of a tensor with multiple delta tensors and triple contraction with
epsilon tensor.
"""

import itertools

import torch
from example.utils import check_one
from torch import Tensor

from carten.reduce import get_permutations_2, symmetrize_and_remove_trace
from carten.utils import check_symmetric_traceless, dij, eijk, letter_index


def get_rules(rank: int, num_delta: int) -> list[str]:
    """
    Rules to contract a tensor with multiple delta tensors and a double contraction with
    epsilon tensor.

    Args:
        rank: the rank of the tensor
        num_delta: the number of delta

    Returns:
        rules, e.g. ['efg,ab,cd,abcdefgh->h'], with deltas, epsilon, and the tensor
        indices in order.
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

        # choose three to contract with epsilon
        for comb in itertools.combinations(remaining, 3):
            epsilon_contracted = "".join(comb)
            uncontracted = "".join(sorted(set(remaining) - set(epsilon_contracted)))

            r = f"{epsilon_contracted},{','.join(delta)},{t_indices}->{uncontracted}"
            rules.append(r)

    return rules


def contract_with_delta_triple_epsilon(
    t: Tensor, num_delta: int, rules: list[str] = None
) -> list[Tensor]:
    """
    Contract a tensor with multiple delta tensors and a triple contraction with
    epsilon tensor.

    Args:
        t: the rank-5 tensor
        num_delta: the number of delta
        rules: the rules to contract

    Returns:
        the contracted tensor
    """
    if rules is None:
        rules = get_rules(t.ndim)

    d = dij(t.device)
    deltas = [d] * num_delta
    e = eijk(t.device)

    out = [torch.einsum(r, e, *deltas, t) for r in rules]

    return out


if __name__ == "__main__":
    ### settings
    torch.manual_seed(30)
    rank = 7
    num_delta = 2
    ###

    t = torch.rand([3] * rank)

    # get the rules
    rules = get_rules(t.ndim, num_delta)
    for i, r in enumerate(rules):
        print("Rule", i, r)

    # get the natural tensors, and checking
    candidates = contract_with_delta_triple_epsilon(t, num_delta, rules)
    natural_tensors = [symmetrize_and_remove_trace(c) for c in candidates]
    for t in natural_tensors:
        check_symmetric_traceless(t)

    # check linear dependence
    for n in range(3, rank + 1):
        # for n in [6]:
        print("=" * 40, "Checking", n)
        check_one(
            natural_tensors,
            rules,
            n,
            factors=[-1, 1],
            atol=1e-5,
            rtol=1e-5,
        )

    print("Done!")
