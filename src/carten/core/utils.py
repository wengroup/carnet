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
    filename: Path = "./H_tensor_and_rule.json.gz", sparse: bool = False
) -> dict[str, dict[str, Tensor]]:
    """
    Load H tensors and the einsum rules for tensor products from a JSON file.

    The H tensors will be converted to torch.Tensor with the default dtype:
    torch.get_default_dtype().

    Args:
        filename: The path to the JSON file containing the H tensors and rules.
        sparse: If `True`, the H tensors will be converted to sparse tensors.


    Returns:
        dict[str, dict[str, Tensor]]: A dictionary containing the H tensors and rules.
    """
    data = json_load(filename)

    # Convert the H tensors from lists to PyTorch tensors
    for key, value in data.items():
        t = torch.tensor(value["H"], dtype=torch.get_default_dtype())

        if sparse:
            l1, l2, l3, _ = key.split("-")
            l1, l2, l3 = int(l1), int(l2), int(l3)
            # Reshape to (3^l3, 3^(l1+l2)) so we can use it on matrix multiplication
            t = t.view(3**l3, 3 ** (l1 + l2)).to_sparse()

        data[key]["H"] = t

    return data
