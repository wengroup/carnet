"""Helper functions to decompose the special tensors or tensor product of two tensors
into natural tensors."""


import warnings

import torch
from torch import Tensor

from carten.natural_tensor import NaturalTensors, remove_trace
from carten.utils import dij, letter_index


def reduce_symmetric_tensor(U: Tensor, start_dim: int = 0) -> NaturalTensors:
    """
    Decompose a fully symmetric tensor into natural tensors.

    Args:
        U: a symmetric tensor
        start_dim: the starting dimension to treat U as a symmetric tensor. Dimensions
            before start_dim will be treated as batch dimensions.

    Returns:
        A NaturalTensors. Let n = U.ndim - start_dim; if n is even, there would be
        n/2 + 1 natural tensors of ranks n, n-2, ..., 2, 0. If n is odd, there would be
        (n+1)/2 natural tensors of rank n, n-2, ..., 3, 1.
    """

    def get_rule(indices: str, num_delta: int):
        delta_indices = ",".join(
            [indices[2 * i : 2 * i + 2] for i in range(0, num_delta)]
        )
        right = indices[num_delta * 2 :]
        return f"{delta_indices}, ...{indices} -> ...{right}"

    n = U.ndim - start_dim
    indices = letter_index(n)
    delta = dij()

    output = [remove_trace(U, start_dim=start_dim)]
    for i in range(1, n // 2 + 1):
        rule = get_rule(indices, i)
        data = [delta] * i + [U]
        symmetrized = torch.einsum(rule, data)
        traceless = remove_trace(symmetrized, start_dim=start_dim)
        output.append(traceless)

    return NaturalTensors.from_sequence(output, start_dim=start_dim)


def get_dyadic_tensor(r: Tensor, rank: int = 2, normalize: bool = True) -> Tensor:
    r"""
    Create a generalized dyadic tensor.

    For rank = 0, the dyadic tensor is a scalar, simply equal to 1.
    For rank = 1, the dyadic tensor is a vector, simply equal to r.
    For rank >= 2, the generalized dyadic tensor is the tensor product of the vector r
    with itself, i.e. :math:`r \otimes r \otimes \cdots \otimes r`. The rank is the
    number of vectors in the tensor product.

    Args:
        r: shape (..., 3) the vector to construct the generalized dyadic tensor. Only
            the last dimension is used to construct the tensor. The ellipsis represents
            any number of dimensions that allows batching.
        rank: rank of the generalized dyadic tensor, i.e. the number of times to tensor
            product the vector r with itself. Rank must be greater than or equal to 1.
        normalize: whether to normalize the vector r as a unit vector before
            constructing the generalized dyadic tensor.

    Returns:
        A tensor of shape (..., 3, 3, ..., 3), where the ... represents the batching
        dimensions, and the number of 3's is equal to the rank.
    """
    if rank < 0:
        raise ValueError("Rank must be greater than or equal to 0.")
    elif rank == 0:
        shape = r.shape[:-1]
        return torch.ones(*shape).to(r.device)
    else:
        if normalize:
            norm = torch.norm(r, dim=-1, keepdim=True)
            if torch.any(norm < 1e-3):
                warnings.warn("The norm of the vector(s) is smaller than 1e-3.")
            r = r / norm

        indices = letter_index(rank)
        data = [r] * rank
        t = torch.einsum(f"{','.join(['...'+i for i in indices])}->...{indices}", *data)

        return t


def reduce_dyadic_tensor(
    r: Tensor, rank: int = 2, normalize: bool = True
) -> NaturalTensors:
    r"""
    Decompose a generalized dyadic tensor into natural tensors.

    Args:
        r: shape (..., 3) the vector to construct the generalized dyadic tensor. The
            ellipsis represents any number of dimensions that allows batching.
        rank: rank of the generalized dyadic tensor, i.e. the number of times to tensor
            product the vector r with itself.
        normalize: whether to normalize the vector r as a unit vector before
            constructing the generalized dyadic tensor.
    """
    U = get_dyadic_tensor(r, rank=rank, normalize=normalize)

    return reduce_symmetric_tensor(U)
