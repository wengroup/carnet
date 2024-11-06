import itertools

import torch
from torch import Tensor


def check_linear_dependence(
    tensors: list[Tensor], factors: list[int], atol: float = 1e-6, rtol: float = 1e-6
) -> tuple[bool, Tensor, Tensor]:
    """
    Check if the tensors are linearly dependent.

    a_0 T_0 + a_1 T_1 + ... + a_n T_n = 0,

    where a_i are the factors.

    Here, all possible combinations of factors are tried for each a_i.

    Args:
        tensors: the tensors to check
        factors: possible non-zero factors, e.g.  [-2, -1, 1, 2]
        atol: absolute tolerance
        rtol: relative tolerance

    Returns:
        linear_dependence: True if the tensors are linearly dependent, False otherwise
        combination: the combination of factors that give the linear dependence
    """
    n = len(tensors)

    tensors = torch.stack(tensors, dim=-1)

    for fac in itertools.product(factors, repeat=n):
        fac = torch.tensor(fac).to(torch.float)
        sum_tensor = torch.sum(fac * tensors, dim=-1)
        if torch.allclose(
            sum_tensor, torch.zeros_like(sum_tensor), atol=atol, rtol=rtol
        ):
            return (
                True,
                fac,
                sum_tensor,
            )
        # else:
        #     print("Combination", fac, 'sum:', sum_tensor)

    return False, None, None


def check_one(
    natural_tensors,
    rules,
    num_selected,
    factors=[-1, 1, -2, 2],
    atol=1e-6,
    rtol=1e-6,
    only_print_dependent=True,
):
    """
    Args:
        natural_tensors: all natural tensors
        rules: all rules
        num_selected: number natural tensors to select to check linear dependence
        factors:  factors to try
    """
    all_selected = itertools.combinations(range(len(natural_tensors)), num_selected)

    for selected_index in all_selected:
        selected_rules = [rules[i] for i in selected_index]
        selected_nts = [natural_tensors[i] for i in selected_index]
        dependent, combs, sums = check_linear_dependence(
            selected_nts, factors, atol=atol, rtol=rtol
        )

        if not only_print_dependent or dependent:
            print("=" * 40)
            print("Linear dependence:", dependent)
            print("Rules:", selected_rules)
            print("Combination factors:", combs)
            print("Combined:", sums)
            if sums is not None:
                mx = sums.abs().max()
            else:
                mx = None
            print("max combined:", mx)
