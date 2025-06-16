from pathlib import Path

import pytest

import carten
from carten.data.dataset import DatasetIP
from carten.data.transform import ConsecutiveAtomType


@pytest.fixture
def dataset():
    filename = Path(carten.__file__).parents[2] / "example" / "dataset" / "SiC.json"
    dataset = DatasetIP(filename=filename, target_names=("energy", "forces"), log=False)

    return dataset


def test_consecutive_atom_type(dataset):
    transform = ConsecutiveAtomType(atomic_number=[14, 6])

    for data in dataset:
        data = transform(data)
        assert set(data.atom_type.tolist()) == {0, 1}

    assert set(transform(dataset).atom_type.tolist()) == {0, 1}
