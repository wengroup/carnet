import torch
from torch import Tensor

from carten.utils import dij, eijk


def find_independent_tensors(tensors: list[Tensor], tolerance=1e-4):
    """Find linearly independent tensors.

    Args:
        tensors: list of tensors
        tolerance: tolerance for checking diagonal elements is non-zero
    """
    vectors = [t.flatten() for t in tensors]
    matrix = torch.vstack(vectors)
    Q, R = torch.linalg.qr(matrix.T, mode="complete")

    # Check all diagonal elements
    independent_indices = []
    for i in range(len(vectors)):
        # TODO, seems not OK to only check diagonal
        if torch.abs(R[i, i]) > tolerance:
            independent_indices.append(i)

    independent_tensors = [tensors[i] for i in independent_indices]

    return independent_tensors, independent_indices


def find_independent_tensors_2(tensors: list[Tensor], tolerance=1e-4):
    """Find linearly independent tensors."""
    vectors = [t.flatten() for t in tensors]
    matrix = torch.vstack(vectors)
    Q, R = torch.linalg.qr(matrix.T, mode="complete")

    print(f"\nAnalysis:")
    print(f"Number of input tensors: {len(tensors)}")

    print("\nQ matrix:")
    print(Q)
    print("\nR matrix:")
    print(R)

    print("\nDiagonal elements:")
    diag = torch.abs(torch.diagonal(R))
    for i, d in enumerate(diag):
        print(f"R[{i},{i}] = {d:.10f}")

    independent_indices = []

    # Check all diagonal elements
    for i in range(len(vectors)):
        # TODO, seems not OK to only check diagonal
        if torch.abs(R[i, i]) > tolerance:
            independent_indices.append(i)
            print(
                f"\nVector {i + 1}:           Independent (diagonal element = {R[i, i]:.10f})"
            )
        else:
            print(f"\nVector {i + 1}: Dependent (diagonal element = {R[i, i]:.10f})")

    independent_tensors = [tensors[i] for i in independent_indices]

    print(f"\nFound {len(independent_indices)} independent tensors")

    return independent_tensors, independent_indices


def tp_with_delta_epsilon(
    t: Tensor, num_delta: int, num_epsilon: int, rule: str
) -> Tensor:
    """
    Contract a tensor with multiple delta tensors and a triple contraction with
    epsilon tensor.

    Args:
        t: the tensor
        num_delta: the number of delta
        rule: the rule to contract

    Returns:
        the contracted tensor
    """
    d = dij(t.device)
    deltas = [d] * num_delta
    e = eijk(t.device)
    eps = [e] * num_epsilon

    return torch.einsum(rule, *eps, *deltas, t)
