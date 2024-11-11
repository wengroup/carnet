import itertools
import string

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
    T: Tensor, atol: float = 1e-8, rtol: float = 1e-5
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

    num_runs = 5

    times = []
    for _ in range(num_runs):
        start = time.time()
        func(*args, **kwargs)
        end = time.time()
        times.append(end - start)

    avg = sum(times) / num_runs
    print(f"Running {func.__name__} for {num_runs} times. Average time: {avg:.6e} s")


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
