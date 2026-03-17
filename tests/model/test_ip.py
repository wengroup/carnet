import torch

from carnet.model.force_stress import compute_forces
from carnet.model.ip import InteratomicPotential


def test_InteratomicPotential(batched_config_info):
    (
        coords,
        atom_type,
        edge_vector,
        edge_idx,
        num_atoms,
        num_atom_types,
    ) = batched_config_info

    F = 5
    max_L = 4

    model = InteratomicPotential(
        F=F,
        max_L=max_L,
        num_atom_types=num_atom_types,
        r_cut=5.0,
        num_layers=2,
        num_average_neigh=1.0,
    )

    energy, e_atom = model(edge_vector, edge_idx, atom_type, num_atoms)

    assert e_atom.shape == (num_atoms.sum(),)

    forces = compute_forces(energy, coords)
    assert forces.shape == (num_atoms.sum(), 3)
    assert torch.allclose(forces[:4], forces[4:])
