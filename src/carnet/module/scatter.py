"""
Scatter operations for tensors.

Based on: https://github.com/rusty1s/pytorch_scatter to avoid installing the package.

Modifications: use torch.scatter_reduce_ instead of torch.scatter_.
"""

from typing import Optional

import torch


def broadcast(src: torch.Tensor, other: torch.Tensor, dim: int):
    if dim < 0:
        dim = other.dim() + dim
    if src.dim() == 1:
        for _ in range(0, dim):
            src = src.unsqueeze(0)
    for _ in range(src.dim(), other.dim()):
        src = src.unsqueeze(-1)
    src = src.expand(other.size())
    return src


def scatter(
    src: torch.Tensor,
    index: torch.Tensor,
    reduce: str,
    dim: int = 0,
    out: Optional[torch.Tensor] = None,
    dim_size: Optional[int] = None,
) -> torch.Tensor:
    """
    A wrapper around torch.scatter_reduce_ that creates output tensor if not provided.

    For "sum" reduction on dim=0 with a 1D index, it uses torch.index_add_ for
    better memory efficiency (avoids broadcasting the index).

    Argues:
        reduce: "sum", "prod", "mean", "amax", "amin".
            Note, "amax"="max" and "amin"="min".
    """
    if dim < 0:
        dim = src.dim() + dim

    if out is None:
        size = list(src.size())
        if dim_size is not None:
            size[dim] = dim_size
        elif index.numel() == 0:
            size[dim] = 0
        else:
            size[dim] = int(index.max()) + 1
        out = torch.zeros(size, dtype=src.dtype, device=src.device)

    # Optimization for sum reduction
    if reduce == "sum" and src.size(dim) == index.size(0):
        return out.index_add_(dim, index, src)

    # For other cases or other reductions, use scatter_reduce_
    # index should have the same shape as src per the PyTorch docs.
    if index.shape != src.shape:
        index = broadcast(index, src, dim)

    return out.scatter_reduce_(dim, index, src, reduce, include_self=False)
