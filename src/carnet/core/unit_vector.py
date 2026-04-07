"""Natural tensors of unit vector.

This implementation is optimized to use precomputed H tensors and flattened
contractions for efficiency.
"""

from pathlib import Path

import torch
from torch import Tensor, nn

from carnet.core.generate_H_unit_vector import load_H_unit_vector


class Polyadics(nn.Module):
    """
    Module to compute natural tensors (polyadics) from unit vectors.

    This module caches the transformation matrices H as a single block-diagonal
    buffer for efficiency.
    """

    def __init__(self, L: int, normalize: str = "unity"):
        super().__init__()
        self.L = L
        self.normalize = normalize

        H_dict = self._get_H_unit_vector(normalize)

        # Combine H tensors for all ranks 0 to L into a single block-diagonal matrix.
        # Rank 0 is [1], Rank 1 is I3.
        # Precompute and store transposed H tensors as a single buffer.
        # Transposing makes the forward contraction efficient: out = a_all @ H_total
        Hs = [torch.ones((1, 1))]
        if L >= 1:
            Hs.append(torch.eye(3))

        for l in range(2, L + 1):
            Hs.append(H_dict[l]["H"].t())

        self.register_buffer("H_total", torch.block_diag(*Hs), persistent=False)

    def forward(self, a: Tensor) -> Tensor:
        """
        Compute all natural tensors of rank 0 to L from unit vectors.

        Args:
            a: Unit vector(s). Shape (..., 3).

        Returns:
            The feature tensor of the unit vector. Shape (..., T), where
            T = (3**(L+1)-1)/2.
        """
        batch_dims = a.shape[:-1]

        # Generate all polyadic tensors (flattened outer products) from rank 0 to L.
        # Rank 0: (..., 1)
        a_flats = [torch.ones(batch_dims + (1,), dtype=a.dtype, device=a.device)]

        # Rank 1: (..., 3)
        if self.L >= 1:
            a_flats.append(a)

        # Ranks 2 to L
        if self.L >= 2:
            curr = a
            for l in range(2, self.L + 1):
                # Flattened outer product: curr = a \otimes ... \otimes a (rank l)
                # (..., 3**(l-1), 1) * (..., 1, 3) -> (..., 3**(l-1), 3) -> flatten
                curr = (curr.unsqueeze(-1) * a.unsqueeze(-2)).flatten(-2)
                a_flats.append(curr)

        # Concatenate all polyadics and perform a single matrix multiplication.
        a_all = torch.cat(a_flats, dim=-1)
        return a_all @ self.H_total

    @classmethod
    def _get_H_unit_vector(cls, normalize: str) -> dict[int, dict[str, any]]:
        """Load precomputed H tensors for unit vectors."""
        filename = Path(__file__).parent / "H_unit_vector.json.gz"
        return load_H_unit_vector(filename, normalize=normalize, mode="flatten")
