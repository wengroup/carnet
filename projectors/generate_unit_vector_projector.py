"""Natural tensors of unit vector.

This implementation is optimized to use precomputed H tensors and flattened
contractions for efficiency.
"""

from pathlib import Path

import torch
from natt.ops import simplify_linear_combination

from carnet.core.unit_vector_2 import get_H, get_H_numerical
from carnet.utils import yaml_dump


def generate_H_unit_vector(max_l: int, dtype: torch.dtype = torch.float64) -> dict:
    """
    Generate H tensors and einsum rules for natural tensors of unit vectors.

    The H tensors are used to transform a polyadic tensor (flattened outer product
    of a unit vector) into a natural tensor.

    Args:
        max_l: Maximum rank of the natural tensors to generate.
        dtype: Data type for the generated tensors. Default is torch.float64.

    Returns:
        dict: A dictionary containing H tensors and rules, keyed by "l-normalize".
    """
    torch.set_default_dtype(dtype)
    all_H = {}

    for l in range(2, max_l + 1):
        # Note, symbolic H is not normalized
        H_symbolic, _, _ = get_H(l)
        H_symbolic = simplify_linear_combination(H_symbolic)

        # replace \u03b4 (delta) by d
        # replace \u03b5 (epsilon) by e
        H_symbolic = str(H_symbolic).replace("\u03b4", "d").replace("\u03b5", "e")

        for normalize in ["unity", "none"]:
            H, rule = get_H_numerical(l, normalize)
            key = f"{l}-{normalize}"
            all_H[key] = {
                "rule": rule,
                "H_symbolic": H_symbolic,
                "H_numerical": H.tolist(),
            }

    return all_H


if __name__ == "__main__":
    max_L = 4
    all_H = generate_H_unit_vector(max_l=max_L)

    # Save to yaml
    filename = Path("./unit_vector_projector.yaml")
    yaml_dump(all_H, filename, compress=True)
