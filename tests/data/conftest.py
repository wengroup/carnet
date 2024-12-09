import shutil
from pathlib import Path

import pytest
from torch_geometric.loader.dataloader import DataLoader

import carten
from carten.data.dataset import Dataset
from carten.data.transform import ConsecutiveAtomType


@pytest.fixture
def dataset():
    shutil.rmtree("processed", ignore_errors=True)

    filename = Path(carten.__file__).parents[2] / "example" / "dataset" / "SiC.json"

    dataset = Dataset(
        filename=filename,
        target_names=("energy", "forces"),
        transform=ConsecutiveAtomType(atomic_number=[14, 6]),
        log=False,
    )

    return dataset


@pytest.fixture
def dataloader(dataset):
    return DataLoader(dataset, batch_size=2, shuffle=False)
