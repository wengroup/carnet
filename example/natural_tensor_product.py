"""Tensor product of two natural tensors.

Here, we use the methods to do it and see that they give the same results.
"""

from itertools import permutations

import torch
from natt.GHS import get_G_H_S
from natt.symmetrize import symmetrize_and_remove_trace
from natt.utils import letter_index
from torch import Tensor

from carten.core.tp import tp_even


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
        l3: rank of the result tensor

    Returns:
        Z: the result tensor
    """
    # Get the G tensor
    perm1 = ["".join(p) for p in permutations(letter_index(l1))]
    perm2 = ["".join(p) for p in permutations(letter_index(l2, start=l1))]

    perms = []
    for p1 in perm1:
        for p2 in perm2:
            p = p1 + p2
            perms.append(p)
    symmetry = "=".join(perms)

    out = get_G_H_S(l1 + l2, symmetry, True)

    # Get the T tensor
    x_letters = letter_index(l1)
    y_letters = letter_index(l2, start=l1)
    T = torch.einsum(f"{x_letters},{y_letters}->{x_letters}{y_letters}", X, Y)

    sum = 0
    for l, GHS in out.items():
        for G, H, S in zip(GHS["G"], GHS["H"], GHS["S"]):
            # Get the X tensor
            X = torch.einsum(H["rule"], H["numerical"], T)

            # Get the T_prime tensor
            T_prime = torch.einsum(G["rule"], G["numerical"], X)

            # Get T prime in another way
            T_prime_2 = torch.einsum(S["rule"], S["numerical"], T)

            assert torch.allclose(
                T_prime, T_prime_2, rtol=1e-5, atol=1e-6
            ), "The two ways to get T' are not the same"

            sum += T_prime

    assert torch.allclose(
        sum, T, rtol=1e-5, atol=1e-6
    ), "sum of T' is not the same as T"

    return out


if __name__ == "__main__":

    l1 = 2
    l2 = 2

    torch.random.manual_seed(35)
    X = torch.randn((3,) * l1)
    X = symmetrize_and_remove_trace(X)
    Y = torch.randn((3,) * l2)
    Y = symmetrize_and_remove_trace(Y)

    out = tp_even_general(X, Y, l1, l2, 2)

    X = X.flatten().unsqueeze(0)
    Y = Y.flatten().unsqueeze(0)

    Z = tp_even(X, Y, l1, l2, 2, normalize="none")
    Z = Z.reshape((3,) * 2)
