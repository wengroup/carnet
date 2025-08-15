import numpy as np
import torch
from ase.io import read

from carten.ase.calculator import CartenCalculator
from carten.utils import get_rotation_matrix

torch.use_deterministic_algorithms(True)


def get_e_f(atoms, idx=None):
    e = atoms.get_potential_energy()
    f = atoms.get_forces()
    if idx is not None:
        f = f[idx]

    return e, f


def test_equivariance():
    #
    # Note, to run this, should generate a water model checkpoint first.
    #

    filename = "./liquid-64.xyz"
    atoms = read(filename)

    atoms.calc = CartenCalculator(
        "/Users/mjwen/Downloads/checkpoint.ckpt", device="cpu"
    )

    e_0, f_0 = get_e_f(atoms)

    # Translation
    atoms.positions += 1
    e_1, f_1 = get_e_f(atoms)

    # Rotation
    R = get_rotation_matrix((10, 20, 30), degrees=True).numpy()
    atoms.cell = atoms.cell.array @ R.T
    atoms.positions = atoms.positions @ R.T
    e_2, f_2 = get_e_f(atoms)

    assert np.allclose(e_0, e_1)
    assert np.allclose(e_0, e_2)

    assert np.allclose(f_0, f_1, atol=1e-4)
    assert np.allclose(f_1 @ R.T, f_2, atol=1e-4)
