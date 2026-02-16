"""
Generate H tensor and rule for tensor product: Z = X \otimes Y.

Z_l3 can be calculated as Z = einsum(rule, H,  X, Y).
"""

from pathlib import Path

import torch
from natt.H_tp import (
    get_H_even,
    get_H_numerical_even,
    get_H_numerical_odd,
    get_H_odd,
    simplify_linear_combination,
)
from torch import Tensor

from carnet.utils import yaml_dump


def generate_H(
    max_l1: int, max_l2: int, max_l3: int, dtype=torch.float64
) -> dict[str : dict[str, Tensor]]:
    """
    Generate H tensors and the corresponding einsum rules for tensor products.

    Args:
        max_l1 (int): Maximum rank of X.
        max_l2 (int): Maximum rank of Y.
        max_l3 (int): Maximum rank of Z.
        dtype: The data type of the generated tensors. Default is torch.float64 for
            higher precision.

    Return:
        H tensors and rules.
        {l1-l2-l3-normalize: {'rule':rule, 'H': H}}
    """
    torch.set_default_dtype(dtype)

    if not max_l1 + max_l2 >= max_l3:
        raise ValueError("l1 + l2 must be greater than or equal to l3")

    # TODO, deal with l1=0 or l2=0 cases
    all_H = {}
    for l1 in range(max_l1 + 1):
        for l2 in range(max_l2 + 1):
            for l3 in range(abs(l1 - l2), min(l1 + l2 + 1, max_l3 + 1)):
                for normalize in ["unity", "none"]:

                    # Even
                    if (l1 + l2 - l3) % 2 == 0:
                        H_symbolic, _, _, _ = get_H_even(l1, l2, l3)
                        H, rule = get_H_numerical_even(l1, l2, l3, normalize)
                    # Odd
                    else:
                        H_symbolic, _, _, _ = get_H_odd(l1, l2, l3)
                        H, rule = get_H_numerical_odd(l1, l2, l3, normalize)

                    H_symbolic = simplify_linear_combination(H_symbolic)
                    # replace \u03b4 (delta) by d
                    # replace \u03b5 (epsilon) by e
                    H_symbolic = (
                        str(H_symbolic).replace("\u03b4", "d").replace("\u03b5", "e")
                    )

                    # Convert to numpy to save it
                    H = H.tolist()
                    key = f"{l1}-{l2}-{l3}-{normalize}"
                    all_H[key] = {
                        "rule": rule,
                        "H_symbolic": H_symbolic,
                        "H_numerical": H,
                    }

    return all_H


if __name__ == "__main__":
    # Generate H tensors and rules
    all_H = generate_H(max_l1=4, max_l2=4, max_l3=4)

    # Save to yaml
    filename = Path("./tensor_product_projector.yaml")
    yaml_dump(all_H, filename, compress=True)
