"""
PyG data point to represent an atomic configuration.
"""
from typing import Optional

import numpy as np
import torch
from pymatgen.core import Structure
from torch import Tensor
from torch_geometric.data import Data

from carten._dtype import DTYPE_BOOL, DTYPE_INT, TORCH_FLOATS, TORCH_INTS
from carten.data.neighbor import get_neigh


class Config(Data):
    """
    Base class of a graph data point (be it a molecule or a crystal).

    Args:
        pos: (num_atoms, 3) array of atomic positions.
        atomic_number: (num_atoms,) array of atomic numbers.
        cell: (3, 3) array of lattice vectors.
        edge_index: (2, num_edges) array of edge indices, where the first row is the
            source node index and the second row is the target node index.
        shift_vec: (num_edges, 3) array of shift vectors. The number of cell boundaries
            crossed by the bond between atom i and j. The distance vector between atom
            j and atom i is given by `pos[j] - pos[i] + shift_vec.dot(cell)`.
        x: input to the model, e.g. initial node features.
        y: reference value for the output of the model, such as energy and forces.
        num_neigh: (num_atoms,) array of the number of neighbors for each atom.
        kwargs: extra property of a data point, e.g. prediction output such as energy
            and forces.

    Notes:
        To use PyG ``InMemoryDataset`` and ``Batch``, all arguments of `Data` should
        be keyword arguments. This is because in collate()
        (see https://github.com/pyg-team/pytorch_geometric/blob/74245f3a680c1f6fd1944623e47d9e677b43e827/torch_geometric/data/collate.py#L14)
         something like `out = cls()` (cls is the subclassed Data class) is used.
        We see that a `Data` class is instantiated without passing arguments.
        For this class, we use ``*`` to distinguish them in the initializer. Arguments
        before ``*`` are necessary (although a default None is provided); arguments
        after it are optional.
    """

    def __init__(
        self,
        pos: np.ndarray = None,
        atomic_number: np.ndarray = None,
        cell: np.ndarray = None,
        edge_index: np.ndarray = None,
        shift_vec: np.ndarray = None,
        x: dict[str, np.ndarray] = None,
        y: dict[str, np.ndarray] = None,
        num_neigh: Optional[np.ndarray] = None,
        **kwargs,
    ):
        ## Should not do the following check, because PyG will create an empty data by
        ## passing None in batching.
        # for name, v in [
        #     ("pos", pos),
        #     ("edge_index", edge_index),
        #     ("shift_vec", shift_vec),
        #     ("x", x),
        #     ("y", y),
        # ]:
        #     if v is None:
        #         raise ValueError(f"None is not allowed for input {name}.")

        DTYPE = torch.get_default_dtype()

        # convert to tensors
        if pos is not None:
            pos = torch.as_tensor(pos, dtype=DTYPE)
            if pos.shape[1] != 3:
                raise ValueError(
                    f"Expect `poss` to be of shape (num_atoms, 3), got {pos.shape}."
                )

            num_atoms = torch.tensor([pos.shape[0]])
        else:
            num_atoms = None

        if atomic_number is not None:
            atomic_number = torch.as_tensor(atomic_number, dtype=DTYPE_INT)

        if cell is not None:
            cell = torch.as_tensor(cell, dtype=DTYPE)
            if cell.shape != (3, 3):
                raise ValueError(
                    f"Expect `cell` to be of shape (3, 3), got {cell.shape}."
                )

        if edge_index is not None:
            edge_index = torch.as_tensor(edge_index, dtype=DTYPE_INT)
            if edge_index.shape[0] != 2:
                raise ValueError(
                    f"Expect `edge_index` to be of shape (2, num_edges), got "
                    f"{edge_index.shape}."
                )
            num_edges = edge_index.shape[1]
        else:
            num_edges = None

        if shift_vec is not None:
            shift_vec = torch.tensor(shift_vec, dtype=DTYPE)
            if shift_vec.shape[0] != num_edges:
                raise ValueError(
                    f"Expect `shift_vec` to be of shape (num_edges, 3), got "
                    f"{shift_vec.shape}."
                )

        if num_neigh is not None:
            num_neigh = torch.as_tensor(num_neigh, dtype=DTYPE_INT)
            if len(num_neigh) != len(pos):
                raise ValueError(
                    f"Expect `num_neigh` to be of length {len(pos)}, got "
                    f"{len(num_neigh)}."
                )

        # convert input and output to tensors
        if x is not None:
            tensor_x = {}
            for k, v in x.items():
                v_out = self._convert_to_tensor(v)
                if v_out is None:
                    raise ValueError(
                        f"Only accepts np.ndarray or torch.Tensor. `{k}` of x is of "
                        f"type `{type(v)}`."
                    )
                tensor_x[k] = v_out
            self._check_tensor_dict(tensor_x, dict_name="x")
        else:
            tensor_x = None

        if y is not None:
            tensor_y = {}
            for k, v in y.items():
                v_out = self._convert_to_tensor(v)
                if v_out is None:
                    raise ValueError(
                        f"Only accepts np.ndarray or torch.Tensor. `{k}` of y is of "
                        f"type `{type(v)}`."
                    )
                tensor_y[k] = v_out
            self._check_tensor_dict(tensor_y, dict_name="y")
        else:
            tensor_y = None

        # convert kwargs to tensor
        tensor_kwargs = {}
        for k, v in kwargs.items():
            v_out = self._convert_to_tensor(v)
            if v_out is None:
                raise ValueError(
                    f"Only accepts np.ndarray or torch.Tensor. kwarg `{k}` is of type "
                    f" `{type(v)}`."
                )
            tensor_kwargs[k] = v_out
        self._check_tensor_dict(tensor_kwargs, dict_name="kwargs")

        super().__init__(
            pos=pos,
            atomic_number=atomic_number,
            cell=cell,
            edge_index=edge_index,
            shift_vec=shift_vec,
            x=tensor_x,
            y=tensor_y,
            num_atoms=num_atoms,
            num_nodes=num_atoms,
            num_neigh=num_neigh,
            **tensor_kwargs,
        )

    # def tensor_property_to_dict(self):
    #     """
    #     Convert all tensor properties to a dict.
    #     """
    #     d = self.to_dict()
    #
    #     out = {}
    #     for k, v in d.items():
    #         if isinstance(v, Tensor):
    #             out[k] = v
    #         elif isinstance(v, dict):
    #             out.update(v)
    #
    #     return out

    @staticmethod
    def _check_tensor_dict(d: dict[str, Tensor], dict_name: str = "name_unknown"):
        """
        Check the values of a dict are at least 1D tensors.

        Args:
            d: the dict to check
            dict_name: name of the dictionary
        """

        for k, v in d.items():
            assert isinstance(
                v, Tensor
            ), f"Expect `{k}` in dict `{dict_name}` to be a tensor, got `{type(v)}`"

            assert len(v.shape) >= 1, (
                f"Expect `{k}` in dict `{dict_name}` to be a tensor at least 1D, "
                f"but its shape is `{v.shape}`."
            )

    @staticmethod
    def _convert_to_tensor(x):
        """Convert a numpy array or torch tensor to a torch tensor.

        If cannot convert, return None.
        """
        DTYPE = torch.get_default_dtype()

        if isinstance(x, np.ndarray):
            if np.issubdtype(x.dtype, np.floating):
                return torch.as_tensor(x, dtype=DTYPE)
            elif np.issubdtype(x.dtype, np.integer):
                return torch.as_tensor(x, dtype=DTYPE_INT)
            elif x.dtype == bool:
                return torch.as_tensor(x, dtype=DTYPE_BOOL)
            else:
                return None

        elif isinstance(x, Tensor):
            if x.dtype in TORCH_FLOATS:
                return torch.as_tensor(x, dtype=DTYPE)
            elif x.dtype in TORCH_INTS:
                return torch.as_tensor(x, dtype=DTYPE_INT)
            elif x.dtype == torch.bool:
                return torch.as_tensor(x, dtype=DTYPE_BOOL)
            else:
                return None

        else:
            return None

    @classmethod
    def from_points(
        cls,
        pos: np.ndarray,
        atomic_number: np.ndarray,
        pbc: bool | tuple[bool, bool, bool],
        cell: np.ndarray | None,
        r_cut: float,
        x: dict[str, np.ndarray],
        y: dict[str, np.ndarray],
        **kwargs,
    ):
        """

        Args:
            pos:
            atomic_number:
            pbc:
            cell: cell vectors. Ignored, if pbc is False, or [False, False, False].
            r_cut:
            x:
            y:
            **kwargs:

        Returns:

        """
        edge_index, shift_vec, num_neigh = get_neigh(
            coords=pos, r_cut=r_cut, pbc=pbc, cell=cell
        )

        if not np.any(pbc):
            cell = None
            shift_vec = None

        return cls(
            pos=pos,
            atomic_number=atomic_number,
            cell=cell,
            edge_index=edge_index,
            shift_vec=shift_vec,
            x=x,
            y=y,
            num_neigh=num_neigh,
            **kwargs,
        )

    @classmethod
    def from_structure(
        cls,
        structure: Structure,
        r_cut: float,
        x: dict[str, np.ndarray],
        y: dict[str, np.ndarray],
        atomic_number: np.ndarray = None,
        pbc=(True, True, True),
        **kwargs,
    ):
        """
        Create a Config from a pymatgen structure.
        """
        return cls.from_points(
            pos=structure.cart_coords,
            atomic_number=(
                atomic_number if atomic_number is not None else structure.atomic_numbers
            ),
            cell=structure.lattice.matrix.copy(),
            pbc=pbc,
            r_cut=r_cut,
            x=x,
            y=y,
            **kwargs,
        )

    @classmethod
    def from_ase(cls, atoms, r_cut: float):
        """
        Create a Config from an ASE atoms object.
        """

        return cls.from_points(
            pos=atoms.positions,
            atomic_number=atoms.get_atomic_numbers(),
            cell=atoms.cell.array.copy(),
            pbc=atoms.pbc,
            r_cut=r_cut,
            x={},
            y={},
        )
