from natt.utils import is_symmetric, is_traceless

from carten.model.tensor_model import AtomicTensorModel, StructureTensorModel


def test_AtomicTensorModel(batched_config_info):
    (
        coords,
        atom_type,
        edge_vector,
        edge_idx,
        num_atoms,
        num_atom_types,
    ) = batched_config_info
    total_num_atoms = num_atoms.sum()

    F = 5
    max_L = 4

    model = AtomicTensorModel(
        F=F,
        max_L=max_L,
        num_atom_types=num_atom_types,
        r_cut=5.0,
        num_layers=2,
        num_average_neigh=1.0,
        output_signature={0: 1, 1: 2, 2: 1},
    )

    atomic_tensor = model(edge_vector, edge_idx, atom_type, num_atoms)
    assert list(atomic_tensor.keys()) == [0, 1, 2]

    assert atomic_tensor[0].shape == (total_num_atoms, 1, 1)

    assert atomic_tensor[1].shape == (total_num_atoms, 2, 3)
    assert is_symmetric(atomic_tensor[1], start_dim=2)
    assert is_traceless(atomic_tensor[1], start_dim=2)

    assert atomic_tensor[2].shape == (total_num_atoms, 1, 9)
    assert is_symmetric(atomic_tensor[2], start_dim=2)
    assert is_traceless(atomic_tensor[2], start_dim=2)


def test_StructureTensorModel(batched_config_info):
    (
        coords,
        atom_type,
        edge_vector,
        edge_idx,
        num_atoms,
        num_atom_types,
    ) = batched_config_info
    num_configs = len(num_atoms)

    F = 5
    max_L = 4

    model = StructureTensorModel(
        F=F,
        max_L=max_L,
        num_atom_types=num_atom_types,
        r_cut=5.0,
        num_layers=2,
        num_average_neigh=1.0,
        output_signature={0: 2, 2: 2, 4: 1},
    )

    structure_tensor = model(edge_vector, edge_idx, atom_type, num_atoms)
    assert list(structure_tensor.keys()) == [0, 2, 4]

    assert structure_tensor[0].shape == (num_configs, 2, 1)

    assert structure_tensor[2].shape == (num_configs, 2, 9)
    assert is_symmetric(structure_tensor[2], start_dim=2)
    assert is_traceless(structure_tensor[2], start_dim=2)

    assert structure_tensor[4].shape == (num_configs, 1, 81)
    assert is_symmetric(structure_tensor[4], start_dim=2)
    assert is_traceless(structure_tensor[4], start_dim=2)
