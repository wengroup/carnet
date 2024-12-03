from fractions import Fraction

import torch
from torch import Tensor


def find_independent_tensors(tensors: list[Tensor], tolerance=1e-4):
    """Find linearly independent tensors using QR decomposition.

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
    """Find linearly independent tensors using QR decomposition."""
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


def matrix_inverse(matrix: list[list[Fraction]]) -> list[list[Fraction]]:
    """
    Calculate the inverse of a matrix containing Fraction objects.
    Returns the inverse matrix with Fraction elements.

    Args:
        matrix: List of lists containing Fraction objects

    Returns:
        Inverse matrix as list of lists with Fraction objects
    """
    n = len(matrix)

    # Create augmented matrix [A|I]
    augmented = []
    for i in range(n):
        row = []
        for j in range(n):
            row.append(matrix[i][j])
        for j in range(n):
            row.append(Fraction(1) if i == j else Fraction(0))
        augmented.append(row)

    # Gaussian elimination
    for i in range(n):
        # Find pivot
        pivot = augmented[i][i]
        if pivot == 0:
            raise ValueError("Matrix is not invertible")

        # Scale row to make pivot 1
        for j in range(2 * n):
            augmented[i][j] = augmented[i][j] / pivot

        # Eliminate column
        for k in range(n):
            if k != i:
                factor = augmented[k][i]
                for j in range(2 * n):
                    augmented[k][j] -= factor * augmented[i][j]

    # Extract inverse matrix
    inverse = []
    for i in range(n):
        inverse.append([])
        for j in range(n):
            inverse[i].append(augmented[i][j + n])

    return inverse
