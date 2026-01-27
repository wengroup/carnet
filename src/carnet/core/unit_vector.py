"""Natural tensors of unit vector.

This implementation is optimized to use precomputed H tensors and flattened
contractions for efficiency.
"""

from pathlib import Path

import torch
from torch import Tensor, nn

from carnet.core.generate_H_unit_vector import load_H_unit_vector
from carnet.module.utils import BufferList


class Polyadics(nn.Module):
    """
    Module to compute natural tensors (polyadics) from unit vectors.

    This module caches the transformation matrices H as buffers for efficiency.
    """

    def __init__(self, L: int, normalize: str = "unity"):
        super().__init__()
        self.L = L
        self.normalize = normalize

        H_dict = self._get_H_unit_vector(normalize)

        # Ranks 0 and 1 are handled specially in forward
        # Store H tensors for ranks l >= 2 as buffers
        Hs = []
        for l in range(2, L + 1):
            # Transpose H for efficient matrix multiplication in forward:
            # n_l = a_flat @ H_l
            Hs.append(H_dict[l]["H"].t())

        self.Hs = BufferList(Hs)

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

        # Rank 0: (..., 1)
        out = [torch.ones(batch_dims + (1,), dtype=a.dtype, device=a.device)]

        # Rank 1: (..., 3)
        if self.L >= 1:
            out.append(a)

        # Ranks 2 to L
        if self.L >= 2:
            a_flat = a
            for l in range(2, self.L + 1):
                # Flattened outer product: a_flat = a \otimes ... \otimes a (rank l)
                # (..., 3**(l-1), 1) * (..., 1, 3) -> (..., 3**(l-1), 3) -> flatten
                a_flat = (a_flat.unsqueeze(-1) * a.unsqueeze(-2)).flatten(-2)

                # N_l = a_flat @ H_l
                H_l = self.Hs[l - 2]
                n_l = a_flat @ H_l
                out.append(n_l)

        return torch.cat(out, dim=-1)

    @classmethod
    def _get_H_unit_vector(cls, normalize: str) -> dict[int, dict[str, any]]:
        """Load precomputed H tensors for unit vectors."""
        filename = Path(__file__).parent / "H_unit_vector.json.gz"
        return load_H_unit_vector(filename, normalize=normalize, mode="flatten")
