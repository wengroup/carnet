import itertools
import string

import torch
from torch import Tensor


def letter_index(n: int, start: int = 0) -> str:
    """
    Get a list of letters 'abc...' of length n.

    Args:
        n: the length of the letters
        start: the starting index.
    """
    return string.ascii_lowercase[start : start + n]


def dij(device: torch.device = None) -> Tensor:
    """Kronecker delta tensor."""
    return torch.eye(3, device=device)


def eijk(device:torch.device=None) -> Tensor:
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
    T: Tensor, start_dim: int = 0, atol: float = 1e-8, rtol: float = 1e-5
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
    T, start_dim: int = 0, atol: float = 1e-8, rtol: float = 1e-5
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
