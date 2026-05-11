from typing import Union

import torch
from torch_geometric.data import Batch
from torch_geometric.transforms import BaseTransform

from carnet._dtype import DTYPE_INT
from carnet.data.data import Config


class ConsecutiveAtomType(BaseTransform):
    """Convert atomic number to consecutive integer atom type.

    For example, for a system of Si and C, their atomic numbers 14 and 6 will be
    converted to atom type 1 and 0, respectively.

    This will modify the input data in-place, by adding a new attribute `atom_type`
    to the data object.
    """

    def __init__(self, atomic_number: list[int], device: torch.device = None):
        """
        Args:
            atomic_number: allowed atom numbers, e.g. [14, 6].
        """

        mapping = -torch.ones(max(atomic_number) + 1, dtype=DTYPE_INT)
        mapping[sorted(atomic_number)] = torch.arange(
            len(atomic_number), dtype=DTYPE_INT
        )
        if device is not None:
            mapping = mapping.to(device)

        self.atomic_number = atomic_number
        self.mapping = mapping

    def forward(self, data: Union[Config, Batch]):
        """
        Return zero-based consecutive atom types.
        """
        atom_type = self.mapping[data.atomic_number]
        if atom_type.min() < 0:
            raise RuntimeError(
                f"Expect atomic numbers to be in {self.atomic_number}, "
                f"got invalid atomic numbers `{data.atomic_number}`."
            )

        data.atom_type = atom_type

        return data
