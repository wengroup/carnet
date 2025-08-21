"""Helper functions to decompose the special tensors or tensor product of two tensors
into natural tensors."""

import torch
from natt.symmetrize import remove_trace
from natt.utils import dij, letter_index
from torch import Tensor

from carnet.legacy.natural_tensor import NaturalTensors

# TODO, this file seems not needed
# Try not using NaturalTensors


def reduce_tensor(
    t: Tensor, symmetry: str = None, start_dim: int = 0
) -> NaturalTensors:
    """
    Decompose a general tensor into natural tensors.

    The reduction spectrum can be determined recursively by the symmetry of the indices.

    Args:
        t: the input tensor
        symmetry: a string that describes the symmetry of the indices. For example,
            `aabb` means the first two indices are symmetric, and next two indices are
            symmetric. Default is None, which means there is no symmetry.
        start_dim: the starting dimension to perform the operation. Dimensions before
            `start_dim` will not be used in the operation.

    Returns:
        A NaturalTensors.
    """


def reduce_symmetric_tensor(u: Tensor, start_dim: int = 0) -> NaturalTensors:
    """
    Decompose a fully symmetric tensor into natural tensors.

    Args:
        u: a symmetric tensor
        start_dim: the starting dimension to perform the operation. Dimensions before
            `start_dim` will not be used in the operation.

    Reference:
        1. http://dx.doi.org/10.1080/00018737800101454

    Returns:
        A NaturalTensors. Let m = u.ndim - start_dim, if n is even, there would be
        m/2 + 1 natural tensors of ranks m, m-2, ..., 2, 0; if n is odd, there would be
        (m+1)/2 natural tensors of rank m, m-2, ..., 3, 1.
    """

    def get_rule(indices: str, num_delta: int):
        delta_indices = ",".join([indices[2 * i : 2 * i + 2] for i in range(num_delta)])
        right = indices[num_delta * 2 :]
        return f"...{indices},{delta_indices}->...{right}"

    m = u.ndim - start_dim  # rank of the tensor
    D = m // 2  # maximum number of deltas that can be contracted

    indices = letter_index(m)
    d = dij(u.device)

    output = [remove_trace(u, start_dim=start_dim)]
    for i in range(1, D + 1):
        rule = get_rule(indices, i)
        v = torch.einsum(rule, u, *([d] * i))
        traceless = remove_trace(v, start_dim=start_dim)
        output.append(traceless)

    return NaturalTensors.from_sequence(output, start_dim=start_dim)
