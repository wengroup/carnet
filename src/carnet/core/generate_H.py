"""
Generate H tensor and rule for tensor product: Z = X \otimes Y.

Z_l3 can be calculated as Z = einsum(rule, H,  X, Y).
"""

from pathlib import Path

import torch
from natt.H_tp import get_H_numerical_even, get_H_numerical_odd
from torch import Tensor

from carnet.utils import json_dump, json_load


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
                        H, rule = get_H_numerical_even(l1, l2, l3, normalize)
                    # Odd
                    else:
                        H, rule = get_H_numerical_odd(l1, l2, l3, normalize)

                    # Convert to numpy to save it
                    H = H.tolist()
                    key = f"{l1}-{l2}-{l3}-{normalize}"
                    all_H[key] = {"rule": rule, "H": H}

    return all_H


def load_H_tensor_and_rule(
    filename: Path = "./H_tensor_and_rule.json.gz", mode: str = None
) -> dict[str, dict[str, Tensor]]:
    """
    Load H tensors and the einsum rules for tensor products from a JSON file.

    The H tensors will be converted to torch.Tensor with the default dtype:
    torch.get_default_dtype().

    Args:
        filename: The path to the JSON file containing the H tensors and rules.
        mode: How to represent the H tensor.
            If None, the H tensor is as is, of shape (3,3,...,3), a total number of
            l3+l2+l1 dimensions.
            If `mm`, the H tensor is reshaped to (3^l3, 3^(l1+l2)) and then transposed
            to (3^(l1+l2), 3^l3) so we can do matrix multiplication with mm(...XY, H).
            if `sparse`, the H tensor is reshaped to (3^l3, 3^(l1+l2)) and then
            transposed to (3^(l1+l2), 3^l3) and converted to a sparse tensor.


    Returns:
        dict[str, dict[str, Tensor]]: A dictionary containing the H tensors and rules.
    """
    data = json_load(filename)

    # Convert the H tensors from lists to PyTorch tensors
    for key, value in data.items():
        # shape (3,3,...,3), a total number of l3+l2+l1 dimensions
        t = torch.tensor(value["H"], dtype=torch.get_default_dtype())

        if mode in ["mm", "sparse"]:
            l1, l2, l3, _ = key.split("-")
            l1, l2, l3 = int(l1), int(l2), int(l3)

            t = t.reshape(3**l3, 3 ** (l1 + l2))

            if mode == "mm":
                # Reshape H to (3^(l1+l2), 3^l3), we we can do mm(...XY, H)
                t = t.transpose(0, 1)
            elif mode == "sparse":
                # We can only do mm(H, XY) since we need gradients w.r.t. X and Y. And
                # sparse tensor do not support gradients.
                t = t.to_sparse()
            else:
                raise ValueError(f"Unknown mode: {mode}")

        elif mode == "flatten":
            l1, l2, l3, _ = key.split("-")
            l1, l2, l3 = int(l1), int(l2), int(l3)

            t = t.reshape(3**l3, 3**l1, 3**l2)

            # Rule, treating all tensor dim as flattened
            rule_new = f"aAB,...A,...B->...a"
            data[key]["rule"] = rule_new

        elif mode is None:
            pass
        else:
            raise ValueError(f"Unknown mode: {mode}")

        data[key]["H"] = t

    return data


if __name__ == "__main__":
    # Generate H tensors and rules
    all_H = generate_H(max_l1=4, max_l2=4, max_l3=4)

    # Save to json
    filename = Path("./H_tensor_and_rule.json")
    json_dump(all_H, filename, compress=True)
