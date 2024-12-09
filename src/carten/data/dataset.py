from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch_geometric.data import InMemoryDataset
from torch_geometric.data.data import BaseData

from carten.data.data import Config


class Dataset(InMemoryDataset):
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
        target_names=("energy", "forces"),  # e.g. (energy, forces, stress)
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
        df = pd.read_json(filename)

        # TODO, this is not used, delete
        if self.atom_featurizer is not None:
            df = self.atom_featurizer(df)

        def process_a_row(row):
            y = {name: np.asarray(row[name]) for name in self.target_names}
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

    def get_mean_atomic_energy(self) -> Tensor:
        """Get the mean atomic energy.

        Returns:
             Scalar tensor.
        """
        energies = []
        for config in self:
            energies.append(config.y["energy"] / len(config["pos"]))

        return torch.tensor(energies).mean()

    def get_root_mean_square_force(self) -> Tensor:
        """Get the root-mean-square force.

        Returns:
             Scalar tensor.
        """
        s = 0
        n = 0
        for config in self:
            s += config.y["forces"].pow(2).sum()
            n += 3 * len(config["pos"])

        rms = (s / n) ** 0.5

        return rms

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


if __name__ == "__main__":
    import carten

    root = Path(carten.__file__).parents[2] / "example" / "dataset"
    filename = root / "SiC.json"
    dataset = Dataset(filename=filename, target_names=("energy", "forces"), log=False)

    for d in dataset:
        print(d)

    y = 1
