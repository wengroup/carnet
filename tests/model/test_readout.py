import torch

from carten.model.readout import AtomicTensor, StructureScalar, StructureTensor


def test_StructureScalar(batched_config_info):
    (
        coords,
        atom_type,
        edge_vector,
        edge_idx,
        num_atoms,
        num_atom_types,
    ) = batched_config_info
    n_configs = len(num_atoms)
    total_num_atoms = num_atoms.sum()

    F = 5
    T = 1
    num_layers = 2

    # create some dummy data
    atom_feats = torch.ones(total_num_atoms, F, T)
    atom_feats = [atom_feats] * num_layers

    module = StructureScalar(num_layers=num_layers, in_features=F, hidden_features=2)
    out = module(atom_feats, atom_type, num_atoms)

    assert out.shape == (n_configs,)


def test_AtomicTensor(batched_config_info):
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
    num_layers = 2
    T = (3 ** (2 + 1) - 1) // 2

    # create some dummy data
    atom_feats = torch.ones(total_num_atoms, F, T)
    atom_feats = [atom_feats] * num_layers

    output_signature = {0: 2, 2: 4}
    module = AtomicTensor(
        num_layers=num_layers,
        in_features=F,
        hidden_features=2,
        output_signature=output_signature,
    )
    out = module(atom_feats, atom_type)

    for l, n in output_signature.items():
        assert out[l].shape == (total_num_atoms, n, 3**l)


def test_StructureTensor(batched_config_info):
    (
        coords,
        atom_type,
        edge_vector,
        edge_idx,
        num_atoms,
        num_atom_types,
    ) = batched_config_info
    n_configs = len(num_atoms)
    total_num_atoms = num_atoms.sum()

    F = 5
    num_layers = 2
    T = (3 ** (2 + 1) - 1) // 2

    # create some dummy data
    atom_feats = torch.ones(total_num_atoms, F, T)
    atom_feats = [atom_feats] * num_layers

    output_signature = {0: 2, 2: 4}
    module = StructureTensor(
        num_layers=num_layers,
        in_features=F,
        hidden_features=2,
        output_signature=output_signature,
    )
    out = module(atom_feats, atom_type, num_atoms)

    for l, n in output_signature.items():
        assert out[l].shape == (n_configs, n, 3**l)
