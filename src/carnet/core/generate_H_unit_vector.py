"""Natural tensors of unit vector.

This implementation is optimized to use precomputed H tensors and flattened
contractions for efficiency.
"""

from pathlib import Path

import torch

from carnet.core.unit_vector_2 import get_H_numerical
from carnet.utils import json_dump, json_load


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
        for normalize in ["unity", "none"]:
            H, rule = get_H_numerical(l, normalize)
            key = f"{l}-{normalize}"
            all_H[key] = {"rule": rule, "H": H.tolist()}

    return all_H


def load_H_unit_vector(
    filename: Path = "./H_unit_vector.json.gz",
    normalize: str = "unity",
    mode: str = None,
) -> dict[int, dict[str, any]]:
    """
    Load precomputed H tensors and rules for unit vectors from a JSON file.

    Args:
        filename: Path to the JSON file containing the H tensors.
        normalize: Normalization method used during generation ("unity" or "none").
        mode: Representation mode. If "flatten", the H tensor is reshaped to
            (3**l, 3**l) and a flattened einsum rule is provided.

    Returns:
        dict[int, dict[str, any]]: A dictionary mapping rank l to another dictionary
            containing the 'H' tensor and the 'rule'.
    """
    data = json_load(filename)
    out = {}
    for key, value in data.items():
        # key format: "l-normalize"
        l_str, norm = key.split("-")
        if norm == normalize:
            l = int(l_str)

            t = torch.tensor(value["H"], dtype=torch.get_default_dtype())
            rule = value["rule"]

            if mode == "flatten":
                if t.ndim != 2:
                    t = t.reshape(3**l, 3**l)
                rule = "ab,...b->...a"

            out[l] = {"H": t, "rule": rule}
    return out


if __name__ == "__main__":
    max_L = 4
    all_H = generate_H_unit_vector(max_l=max_L)

    filename = Path(__file__).parent / "H_unit_vector.json"
    json_dump(all_H, filename, compress=True)
    print(
        f"Generated H tensors for unit vector up to rank {max_L} and saved to {filename}.gz"
    )
