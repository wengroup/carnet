import itertools
import string
from fractions import Fraction

import torch
from torch import Tensor


def letter_index(n: int, start: int = 0, upper_case: bool = False) -> str:
    """
    Get a list of letters 'abc...' of length n.

    Args:
        n: the length of the letters
        start: the starting index.
        upper_case: whether to use upper case letters.
    """
    if upper_case:
        return string.ascii_uppercase[start : start + n]
    else:
        return string.ascii_lowercase[start : start + n]


def double_index(n: int, start: int = 0) -> list[str]:
    """
    Get multiple double indices, like ['ab', 'cd', 'ef'].

    Args:
        n: the number of double indices
        start: the starting index

    Examples:
        >>> double_index(2)
        ['ab', 'cd']
        >>> double_index(3, start=1)
        ['bc', 'cd', 'de']
    """
    indices = letter_index(2 * n, start)
    return [indices[i : i + 2] for i in range(0, 2 * n, 2)]


def repeat_double_index(n: int, start: int = 0) -> list[str]:
    """
    Get multiple repeated double indices, like ['aa', 'bb', 'cc'].

    Args:
        n: the number of double indices
        start: the starting index

    Examples:
        >>> repeat_double_index(2)
        ['aa', 'bb']
        >>> repeat_double_index(3, start=1)
        ['bb', 'cc', 'dd']
    """
    indices = letter_index(n, start)
    return [s * 2 for s in indices]


def dij(device: torch.device = None) -> Tensor:
    """Kronecker delta tensor."""
    return torch.eye(3, device=device)


def eijk(device: torch.device = None) -> Tensor:
    """Levi-Civita tensor."""
    e = torch.zeros(3, 3, 3, device=device)
    e[0, 1, 2] = 1.0
    e[1, 2, 0] = 1.0
    e[2, 0, 1] = 1.0
    e[0, 2, 1] = -1.0
    e[1, 0, 2] = -1.0
    e[2, 1, 0] = -1.0

    return e


def get_trace(T: Tensor, i: int, j: int) -> Tensor:
    """
    Trace of a tensor between two indices.

    Args:
        T: input tensor
        i: first index
        j: second index

    Example:
        T_ijkl -> T_ijil
    """

    assert i < T.ndim and j < T.ndim, "Index out of range"

    indices = letter_index(T.ndim)
    rule = indices.replace(indices[j], indices[i])
    trace = torch.einsum(rule, T)

    return trace


def check_symmetric(
    T: Tensor, start_dim: int = 0, atol: float = 1e-6, rtol: float = 1e-5
) -> bool:
    """
    Check if a tensor is fully symmetric.

    Args:
        T: input tensor
        start_dim: the starting dimension to check symmetry
    """

    if T.ndim - start_dim <= 1:
        return True

    for p in itertools.permutations(range(start_dim, T.ndim)):
        p = list(range(start_dim)) + list(p)
        permuted = T.permute(*p)
        if not torch.allclose(T, permuted, atol=atol, rtol=rtol):
            e = T - permuted
            error = torch.sum(torch.abs(e))
            return False

    return True


def check_traceless(
    T, start_dim: int = 0, atol: float = 1e-6, rtol: float = 1e-5
) -> bool:
    """Check if a tensor is traceless.

    Args:
        T: input tensor
        start_dim: the starting dimension to check tracelessness
    """

    rank = T.ndim - start_dim

    if rank <= 1:
        return True
    elif rank == 2:
        zeros = torch.tensor(0.0)
    else:
        dims = [3] * (rank - 2)
        zeros = torch.zeros(*dims)

    for i, j in itertools.combinations(range(start_dim, T.ndim), 2):
        trace = get_trace(T, i, j)
        if not torch.allclose(trace, zeros, atol=atol, rtol=rtol):
            return False

    return True


def check_symmetric_traceless(
    T: Tensor, atol: float = 1e-6, rtol: float = 1e-5
) -> bool:
    """Check if a tensor is symmetric and traceless."""
    return check_symmetric(T, atol=atol, rtol=rtol) and check_traceless(
        T, atol=atol, rtol=rtol
    )


def check_shape(T: Tensor, n: int = 3) -> bool:
    """Check a tensor is of shape (n, n, ..., n)"""
    if T.ndim == 0:
        return True
    elif set(T.shape) != {n}:
        return False
    else:
        return True


def time_it(func, *args, **kwargs):
    """Time a function."""
    import time

    num_runs = 1

    times = []
    for _ in range(num_runs):
        start = time.time()
        out = func(*args, **kwargs)
        end = time.time()
        times.append(end - start)

    avg = sum(times) / num_runs
    print(f"Running {func.__name__} for {num_runs} times. Average time: {avg:.6e} s")

    return out


def factorial(n: int, device: torch.device = None):
    """
    Get the factorial of a number.
    """
    return torch.prod(torch.arange(1, n + 1, device=device))


def double_factorial(
    n: int, lower_bound: int = None, device: torch.device = None
) -> Tensor:
    """
    Get the double factorial of a number.

    Args:
        n: The number to calculate the double factorial
        lower_bound: The lower bound of the double factorial. If lower bound is
            provided, this is calculated as n * (n-2) * ... * lower_bound. Default is
            None, meaning 1 if n odd and 2 if n even.
    """

    if n == 0 or n == 1:
        return torch.tensor(1, device=device)
    elif n % 2 == 0:
        if lower_bound is None:
            lower_bound = 2
        else:
            assert lower_bound % 2 == 0, "lower_bound must be even"
        return torch.prod(torch.arange(lower_bound, n + 2, step=2, device=device))
    else:
        if lower_bound is None:
            lower_bound = 1
        else:
            assert lower_bound % 2 == 1, "lower_bound must be odd"
        return torch.prod(torch.arange(lower_bound, n + 2, step=2, device=device))


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
