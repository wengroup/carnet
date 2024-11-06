"""
Find the linear dependence rules of double contracting epsilon tensor with multiple
tensors obtained from a parent general T.
"""

import itertools

import torch
from torch import Tensor
from utils import check_one

from carten.reduce import symmetrize_and_remove_trace
from carten.utils import check_symmetric_traceless, eijk


def get_rules(rank: int) -> list[str]:
    """
    Get the contraction rule of double contracting epsilon tensor with a tensor.

    epsilon_{abc} T_bcd...

    Args:
        rank: rank of the tensor

    Returns:
        list of contraction rules
    """

    rules = []
    for comb in itertools.combinations(range(rank), 2):
        Gcandidate = ["" for _ in range(rank)]

        # choose two positions to contract
        pos_b, pos_c = comb
        candidate[pos_b] = "b"
        candidate[pos_c] = "c"

        # Fill remaining positions with d, e, f, ...
        next_char = ord("d")
        for i in range(rank):
            if i not in comb:
                candidate[i] = chr(next_char)
                next_char += 1

        left = "".join(candidate)
        right = left.replace("b", "").replace("c", "")

        rules.append(f"abc,{left}->a{right}")

    return rules


def contract_double_epsilon(t: Tensor, rules: list[str] = None) -> list[Tensor]:
    """
    Double contract epsilon tensor with a tensor.

    Args:
        t: the tensor to contract
        rules: rules for contraction. If None, generate all unique rules and then use.

    Returns:
        list of contracted tensors
    """

    if rules is None:
        rules = get_rules(t.ndim)

    e = eijk(t.device)

    out = [torch.einsum(r, e, t) for r in rules]

    return out


if __name__ == "__main__":
    #### settings
    torch.manual_seed(30)
    rank = 4
    ###

    t = torch.rand([3] * rank)

    # contraction rules
    rules = get_rules(rank)
    for i, r in enumerate(rules):
        print("Rule", i, r)

    candidates = contract_double_epsilon(t, rules)

    # get natural tensors, and check
    natural_tensors = [symmetrize_and_remove_trace(c) for c in candidates]
    for t in natural_tensors:
        check_symmetric_traceless(t)

    # check linear dependence
    for n in range(3, rank + 1):
        print("=" * 60, "Checking num components:", n)
        check_one(
            natural_tensors,
            rules,
            n,
            factors=[-1, 1],
            atol=1e-5,
            rtol=1e-5,
        )

    print("Done!")
