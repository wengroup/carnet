from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LinearRegression
from torch import Tensor
from torch_geometric.data import InMemoryDataset
from torch_geometric.data.data import BaseData

from carnet.data.data import Config


class BaseDataset(InMemoryDataset):
    """
    A base dataset to read atomic data for both structures and molecules.

    This input dataset is a pandas DataFrame file, e.g. a json file.

    The below columns are mandatory:
        - `coords`, coordinates of the atoms
        - `atomic_number`, atomic number of the atoms in the periodic table
        - `energy`, total energy of the system

    The below columns are optional:
        - `cell`, supercell vectors of the system; this is typically used for periodic
          crystals, and not needed for molecules.
        - `pbc`, periodic boundary conditions of the system; this is typically used for
          crystals, and not needed for molecules.
        - `forces`, forces on the atoms
        - `stress`, stress tensor of the system
    """

    def __init__(
        self,
        filename: Path,
        target_names: list[str],
        r_cut: float = 5.0,
        atom_featurizer: Callable = None,
        root=".",
        transform=None,
        pre_transform=None,
        pre_filter=None,
        log: bool = True,
    ):
        """
        Args:
            filename: path to pandas DataFrame file, e.g. a json file.
            r_cut: cutoff radius.
            target_names: names of the targets (e.g. energy, forces, stress) that will
                be extracted from the dataframe and stored in the data object.
            atom_featurizer: a callable that featurize atoms, e.g. convert atomic number
                to numerical values. If None, the atomic number is used.
        """
        self.filename = filename
        self.target_names = target_names
        self.r_cut = r_cut
        self.atom_featurizer = atom_featurizer

        super().__init__(root, transform, pre_transform, pre_filter, log=log)
        self.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> str | list[str] | tuple[str, ...]:
        return self.filename

    @property
    def processed_file_names(self) -> str | list[str] | tuple[str, ...]:
        path = Path(self.filename).expanduser().resolve()
        filename = path.name
        name = filename.replace(path.suffix, "_processed.pt")

        return name

    def process(self):
        data_list = self.read_data(self.raw_paths[0])

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        self.save(data_list, self.processed_paths[0])

    def read_data(self, filename: str) -> list[Config]:
        """Read data from the file and return a list of Config objects."""
        raise NotImplementedError

    def get_atomic_number(self) -> list[int]:
        """Get all atomic number of the atoms in the data."""
        atom_numbers = set(self.atomic_number.tolist())
        return sorted(atom_numbers)

    def get_num_average_neigh(self) -> Tensor:
        """Number of neighbors for each atom averaged across the dataset.

        Returns:
             Scalar tensor.
        """
        return self.num_neigh.to(torch.get_default_dtype()).mean()

    @classmethod
    def save(cls, data_list: Sequence[BaseData], path: str) -> None:
        r"""Saves a list of data objects to the file path :obj:`path`."""

        from torch_geometric.io import fs

        data, slices = cls.collate(data_list)
        # this below line is used in torch_geometric>2.4.0
        # fs.torch_save((data.to_dict(), slices, data.__class__), path)

        # The below modification makes it go to the original implementation of
        # torch_geometric<=2.4.0
        # This will use `Data` instead of `Config` to restore the save dataset it in
        # the `load` method. This ca avoid shape mismatch error we have enforced in
        # the `Config` class.
        fs.torch_save((data.to_dict(), slices), path)


class DatasetIP(BaseDataset):
    """Dataset for interatomic potentials."""

    def read_data(self, filename: str) -> list[Config]:
        df = pd.read_json(filename)

        def process_a_row(row):
            y = {name: row[name] for name in self.target_names}
            y = _to_numpy(y)

            y["energy"] = y["energy"].reshape(1)

            if "cell" not in row:
                cell = None
            else:
                cell = np.asarray(row["cell"])

            if "pbc" not in row:
                pbc = False
            else:
                pbc = row["pbc"]

            data = Config.from_points(
                pos=np.asarray(row["coords"]),
                atomic_number=np.asarray(row["atomic_number"]),
                pbc=pbc,
                cell=cell,
                r_cut=self.r_cut,
                x=None,
                y=y,
            )

            return data

        configs = df.apply(process_a_row, axis=1).tolist()

        return configs

    def get_mean_atomic_energy(self) -> Tensor:
        """Get the mean atomic energy.

        Returns:
             Scalar tensor.
        """
        return _get_mean_atomic_energy(self)

    def get_linear_fit_atomic_energy(self) -> Tensor:
        """Get the atomic energy per element by linear fitting."""

        atomic_numbers = [config["atomic_number"].tolist() for config in self]
        energies = [config.y["energy"].item() for config in self]

        return linear_fit_atomic_energy(atomic_numbers, energies)

    def get_root_mean_square_force(self) -> Tensor:
        """Get the root-mean-square force.

        Returns:
             Scalar tensor.
        """
        return _get_root_mean_square_force(self)


class DatasetTensor(BaseDataset):
    """Dataset for tensor property prediction."""

    def read_data(self, filename: str) -> list[Config]:
        df = pd.read_json(filename)

        def process_a_row(row):
            y = {name: row[name] for name in self.target_names}
            y = _to_numpy(y)

            # TODO, this is hard coded for `full` and `voigt` tensors
            for k, v in y.items():
                if k.endswith("_full"):

                    # TODO, this is hard coded for `atomic tensor`, we do not need a
                    #   new dim for batching.
                    if "shielding" in k:
                        y[k] = v
                    else:
                        # Add additional dim for batching for `structure tensor`.
                        y[k] = np.expand_dims(v, 0)

                if k.endswith("_voigt"):
                    # Add additional dim for batching, and flatten the tensor dim
                    # We need the flattening for metrics computing, where the
                    # prediction has flattened tensor dim.
                    y[k] = v.reshape(1, -1)

            if "cell" not in row:
                cell = None
            else:
                cell = np.asarray(row["cell"])

            if "pbc" not in row:
                pbc = False
            else:
                pbc = row["pbc"]

            data = Config.from_points(
                pos=np.asarray(row["coords"]),
                atomic_number=np.asarray(row["atomic_number"]),
                pbc=pbc,
                cell=cell,
                r_cut=self.r_cut,
                x=None,
                y=y,
            )

            return data

        configs = df.apply(process_a_row, axis=1).tolist()

        return configs

    def get_shift_and_scale_tensors(
        self,
    ) -> tuple[dict[int, Tensor], dict[int, Tensor]]:
        """Get the shift and scale of tensors of natural tensors.

        Shift is only defined for rank-0 tensors, and it is computed as the mean of
        the rank-0 tensors.

        Scale are computed for all tensors. For rank-0 tensor, it is the standard
        deviation. For tensors of rank > 0, it is the root-mean-square of the tensor
        elements.

        Tensors in self.y of different ranks all have the same shape: (B, F, T), where:
        `B` is batch dimension, which is always 1 for all the tensors.
        `F` number of natural tensors (namely seniority) of the tensor decomposition.
        `T` is the number of tensor elements, and T = 3^l, where l is the rank of the
        natural tensor.
        For example, an elastic tensor can be decomposed to two rank-0 natural tensors,
        two rank-2 natural tensors, and one rank-4 natural tensor. Then F=2,
        T = 1 for rank-0 tensors; F=2, T = 9 for rank-2 tensors; F=1, T=81 for rank-4
        tensors.

        In this function, we compute the shift/scale separately for each dim in `F`,
        and the statistics are computed from the `T` dimension.

        Returns:
            shifts: {rank, value}. Currently, only rank-0 tensors have shift. The shape
                of the value tensor is  (F, 1).
            scales: {rank: value}. The shape of the value tensor is (F, 1).
        """
        # TODO, this is hard coded, we get the natural tensor by name search
        target_names = [n for n in self.target_names if n.endswith("_natural")]
        if len(target_names) != 1:
            raise RuntimeError("Only one target name is allowed.")
        else:
            name = self.target_names[0]

        return _get_tensor_shifts_and_scales(self, name)


class DatasetMultiTask(DatasetTensor):
    """
    An extension of DatasetTensor to handle multiple tensor targets.
    """

    def get_shift_and_scale_tensors(
        self,
    ) -> tuple[dict[str, dict[int, Tensor]], dict[str, dict[int, Tensor]]]:

        shifts = {}
        scales = {}

        # energy
        shifts["energy"] = _get_mean_atomic_energy(self)
        scales["energy"] = _get_root_mean_square_force(self)

        # TODO, this is hard coded, we get the natural tensor by name search
        target_names = [n for n in self.target_names if n.endswith("_natural")]

        for name in target_names:
            s, c = _get_tensor_shifts_and_scales(self, name)
            shifts[name] = s
            scales[name] = c

        return shifts, scales


def _to_numpy(d):
    """
    Convert values to numpy arrays.

    The values can be given as dictionary values, and the dictionary can be nested.
    """

    if isinstance(d, dict):
        return {k: _to_numpy(v) for k, v in d.items()}
    else:
        return np.asarray(d)


def _get_mean_atomic_energy(self) -> Tensor:
    energies = []
    for config in self:
        energies.append(config.y["energy"] / len(config["pos"]))

    return torch.tensor(energies).mean()


def _get_root_mean_square_force(self) -> Tensor:
    s = 0
    n = 0
    for config in self:
        s += config.y["forces"].pow(2).sum()
        n += 3 * len(config["pos"])

    rms = (s / n) ** 0.5

    return rms


def linear_fit_atomic_energy(
    atomic_number: list[list[int]], energy: list[float]
) -> Tensor:
    """
    Perform a linear fit to get the atomic energy per element.

    Args:
        atomic_number: list of atomic numbers for each configuration.
        energy: total energy for each configuration.

    Returns:
        Atomic energy per element. The shape is (N+1,), where N max atomic number in the
        dataset. The 0-th element is unused. As such, atomic_energy[Z] gives the atomic
        energy for element with atomic number Z.
    """
    # Get unique atomic numbers and create mapping between atomic number and index
    atomic_number_set = set().union(*(set(am) for am in atomic_number))
    idx_to_atomic_number = {i: z for i, z in enumerate(sorted(atomic_number_set))}
    atomic_number_to_idx = {v: k for k, v in idx_to_atomic_number.items()}

    # Prepare feature matrix X, where X[i, j] is the count of element j in config i.
    # Note, j is the index, not the atomic number
    X = []
    for am in atomic_number:
        indices = [atomic_number_to_idx[z] for z in am]
        x = np.bincount(indices, minlength=len(atomic_number_set))
        X.append(x)

    # Fit X*a = y to get atomic energy, intercept should not be used
    model = LinearRegression(fit_intercept=False)
    model.fit(X, energy)

    # Convert to tensor with shape (max_atomic_number + 1,)
    atomic_energy = torch.zeros(max(atomic_number_set) + 1)
    for i, e in enumerate(model.coef_):
        z = idx_to_atomic_number[i]
        atomic_energy[z] = e

    return atomic_energy


def _get_tensor_shifts_and_scales(self, name):
    # Note, the key of y[name] is an integer that is represented as a string
    rank_0_vals = []
    scales = {rank: 0 for rank, _ in self[0].y[name].items() if rank != "0"}

    for config in self:
        d = config.y[name]
        for rank, val in d.items():
            assert val.ndim == 3, "The tensor should have 3 dimensions: (1, F, T)."
            if rank == "0":
                rank_0_vals.append(val)
            else:
                # sum over F, and T dims
                scales[rank] += val.pow(2).sum()

    # Scale of tensors of rank > 0
    # normalize each of scales by N*F*T (which is the total number of components summed
    # in each of scales), where:
    # - N is the number of configurations;
    # - F is the number of natural tensors; Typically it is 1 for structure tensors,
    #   and it can be > 1 for atomic tensors where each configuration consists of
    #   multiple atomic tensors.
    # - T is the number of tensor elements, which is 3^l, where l is the rank of the
    #   natural tensor.
    scales = {
        int(rank): (v / (len(self) * val.shape[1] * 3 ** int(rank))).sqrt()
        for rank, v in scales.items()
    }

    # Scale and shift of rank 0 tensors
    if rank_0_vals:
        rank_0_vals = torch.cat(rank_0_vals)
        scales[0] = torch.std(rank_0_vals)
        shifts = {0: torch.mean(rank_0_vals)}
    else:
        shifts = {}

    return shifts, scales


if __name__ == "__main__":
    import carnet

    root = Path(carnet.__file__).parents[2] / "example" / "dataset"
    filename = root / "SiC.json"
    dataset = DatasetIP(filename=filename, target_names=["energy", "forces"], log=False)

    for d in dataset:
        print(d)

    y = 1
