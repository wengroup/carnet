"""Tensor product of two natural tensors.

Here, we use the methods to do it and see that they give the same results.

One method is using the explicit rule and another is using the general rule. We want to
explore the scalar factor between the two methods.
The two methods are:
1. get_G_H_S_of_j_natural: the general method, where G is obtained using the tensor
   decomposition method.
2. tp_even: the explicit method, where the G tensor is from the explicit tensor rule.
"""

import torch
from natt.evaluate import evaluate_tensors
from natt.GHS import get_G_H_S_of_j_natural
from natt.symmetrize import get_random_natural_tensor
from natt.utils import letter_index
from torch import Tensor

from carnet.core.tp import tp_even


def tp_even_general(X: Tensor, Y: Tensor, l1: int, l2: int, l3: int) -> Tensor:
    """
    The general method to get the product of natural tensors.

    1. Get T = X \otimes Y
    2. Use the G tensor to get

    Args:
        X: the first natural tensor
        Y: the second natural tensor
        l1: rank of the first tensor
        l2: rank of the second tensor

    Returns:
        Z: the result tensor
    """

    G, _, _, _, _ = get_G_H_S_of_j_natural(l1, l2, l3)
    G = evaluate_tensors(G, mode="G")

    x_idx = letter_index(l1)
    y_idx = letter_index(l2, start=l1)
    z_idx = letter_index(l3, upper_case=True)
    rule = f"{z_idx}{x_idx}{y_idx},{x_idx},{y_idx}->{z_idx}"

    Z = torch.einsum(rule, G, X, Y)

    return Z


if __name__ == "__main__":
    l1 = 3
    l2 = 3
    l3 = 2
    X = get_random_natural_tensor(l1, seed=1)
    Y = get_random_natural_tensor(l2, seed=2)

    # General method
    Z1 = tp_even_general(X, Y, l1, l2, l3)

    # Explicit method
    NORMALIZATION = "unity"  # or "none"
    X = X.flatten().unsqueeze(0)
    Y = Y.flatten().unsqueeze(0)
    Z2 = tp_even(X, Y, l1, l2, l3, normalize=NORMALIZATION)
    Z2 = Z2.reshape((3,) * l3)

    # Factor between the two methods
    # We will see that no  matter what the normalization is, the factor will not be 1.
    factor = Z1 / Z2

    print(factor)
