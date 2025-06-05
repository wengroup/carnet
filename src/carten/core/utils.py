from pathlib import Path

import torch
from torch import Tensor

from carten.utils import json_load


def check_shape(T: Tensor, n: int = 3) -> bool:
    """Check a tensor is of shape (n, n, ..., n)"""
    if T.ndim == 0:
        return True
    elif set(T.shape) != {n}:
        return False
    else:
        return True


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

            rule = value["rule"]

            if l1 != 0:
                r_l1 = "A"
            else:
                r_l1 = ""
            if l2 != 0:
                r_l2 = "B"
            else:
                r_l2 = ""
            if l3 != 0:
                r_l3 = "a"
            else:
                r_l3 = ""

            rule_new = f"{r_l3}{r_l1}{r_l2},...{r_l1},...{r_l2}->...{r_l3}"

            data[key]["rule"] = rule_new

        elif mode is None:
            pass
        else:
            raise ValueError(f"Unknown mode: {mode}")

        data[key]["H"] = t

    return data
