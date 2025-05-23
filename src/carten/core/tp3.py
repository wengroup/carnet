"""
Same as tp2.py, but:
- use matrix multiplication instead of einsum
"""

from pathlib import Path

import torch
from line_profiler import profile
from natt.H_tp import get_H_numerical_even, get_H_numerical_odd
from torch import Tensor

from carten.core.utils import load_H_tensor_and_rule

# Load the pre-computed H tensor and einsum rule
filename = Path(__file__).parent / "H_tensor_and_rule.json.gz"
H_TENSOR_AND_RULE = load_H_tensor_and_rule(filename, mode="mm")

# On device cache for efficiency
H_TENSOR_AND_RULE_ON_DEVICE = set()


@profile
def tp_even(
    X: Tensor, Y: Tensor, l1: int, l2: int, l3: int, normalize: str = "unity"
) -> Tensor:
    """
    Calculate the tensor product Z_l3 = X_l1 \otimes Y_l2 where l1 + l2 - l3 is even.

    Args:
        X: A natural tensor of rank l1. Shape: (..., F, 3^l1), where F is the number of
            features.
        Y: A natural tensor of rank l2. Shape: (..., F, 3^l2), where F is the number of
            features.
        l1: The rank of the first tensor X.
        l2: The rank of the second tensor Y.
        l3: The rank of the output tensor Z.
        normalize: The normalization method.
            If `unity`, the output is normalized such that the l3 fold contraction of
            the output tensor with a unit vector yields 1.
            If `none`, no normalization is applied.

    Returns:
        A natural tensor of rank l3. Shape: (..., F, 3^l3), where F is the number of
        features.
    """
    assert abs(l1 - l2) <= l3 <= l1 + l2, "l3 must be in the range of |l1-l2| and l1+l2"
    assert (l1 + l2 - l3) % 2 == 0, "l1 + l2 - l3 must be even"

    leading_dims = X.shape[:-1]  # including the feature dimension

    # Get H tensor and einsum rule:
    # H, rule = get_H_numerical_even(l1, l2, l3, normalize)
    # We use the pre-computed H tensor and rule for efficiency
    #
    # Shape of H: (3^(l1+l2), 3^l3)
    H, _ = get_H_and_rule(l1, l2, l3, normalize, X.device)

    XY = torch.einsum("...x,...y->...xy", X, Y)  # (..., F, 3^l1, 3^l2))
    XY = XY.reshape(*leading_dims, 3 ** (l1 + l2))

    Z = torch.matmul(XY, H)  # (leading_dims, 3**l3)

    return Z


@profile
def tp_odd(
    X: Tensor, Y: Tensor, l1: int, l2: int, l3: int, normalize: str = "unity"
) -> Tensor:
    """
    Calculate the tensor product Z_l3 = X_l1 \otimes Y_l2 where l1 + l2 - l3 is odd.

    Args:
        X: A natural tensor of rank l1. Shape: (..., F, 3^l1), where F is the number of
            features.
        Y: A natural tensor of rank l2. Shape: (..., F, 3^l2), where F is the number of
            features.
        l1: The rank of the first tensor X.
        l2: The rank of the second tensor Y.
        l3: The rank of the output tensor Z.

    Returns:
        A natural tensor of rank l3. Shape: (..., F, 3^l3), where F is the number of
        features.
    """
    assert abs(l1 - l2) <= l3 <= l1 + l2, "l3 must be in the range of |l1-l2| and l1+l2"
    assert (l1 + l2 - l3) % 2 == 1, "l1 + l2 - l3 must be odd"

    leading_dims = X.shape[:-1]  # including the feature dimension

    # Get H tensor and einsum rule:
    # H, rule = get_H_numerical_odd(l1, l2, l3, normalize)
    # We use the pre-computed H tensor and rule for efficiency
    #
    # Shape of H: (3^(l1+l2), 3^l3)
    H, _ = get_H_and_rule(l1, l2, l3, normalize, X.device)

    XY = torch.einsum("...x,...y->...xy", X, Y)  # (..., F, 3^l1, 3^l2))
    XY = XY.reshape(*leading_dims, 3 ** (l1 + l2))

    Z = torch.matmul(XY, H)  # (leading_dims, 3**l3)

    return Z


def get_H_and_rule(
    l1: int, l2: int, l3: int, normalize: str, device: torch.device = None
):

    key = f"{l1}-{l2}-{l3}-{normalize}"

    global H_TENSOR_AND_RULE_ON_DEVICE
    if key not in H_TENSOR_AND_RULE_ON_DEVICE:
        try:
            H_and_rule = H_TENSOR_AND_RULE[key]
        except KeyError:
            raise RuntimeError(
                f"Pre-computed H tensor and einsum rule not found for {key}."
                "You can generate them using the `generate_H.py` file."
            )

        # Move to device
        H_and_rule["H"] = H_and_rule["H"].to(device)
        H_TENSOR_AND_RULE_ON_DEVICE.add(key)

    # Get H and einsum rule
    H = H_TENSOR_AND_RULE[key]["H"]
    rule = H_TENSOR_AND_RULE[key]["rule"]

    return H, rule


if __name__ == "__main__":
    from natt.symmetrize import get_random_natural_tensor

    l1 = 4
    l2 = 4
    l3 = 4
    l4 = 3

    X = get_random_natural_tensor(l1, seed=1).view(-1)
    Y = get_random_natural_tensor(l2, seed=2).view(-1)

    tp_even(X, Y, l1, l2, l3)
    tp_odd(X, Y, l1, l2, l4)
