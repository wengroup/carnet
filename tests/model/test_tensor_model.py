import torch
from lightning import seed_everything
from natt.utils import is_symmetric, is_traceless

from carnet.core.convert import Converter
from carnet.model.tensor_model import AtomicTensorModel, StructureTensorModel
from carnet.utils import get_rotation_matrix


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


def test_rotation(config_info):
    """
    Test the equivariance of the StructureTensorModel.
    R f(x) = f(R x)

    """
    _, atom_type, edge_vector, edge_idx, num_atoms, num_atom_types = config_info

    # Rotate the coordinates and lattice vectors
    R = get_rotation_matrix((30, 45, 60), degrees=True)
    rotated_edge_vector = edge_vector @ R.T

    converter = Converter(symmetry="ijkl=jikl=klij")

    # Elastic tensor
    model = StructureTensorModel(
        F=6,
        max_L=4,
        num_atom_types=num_atom_types,
        r_cut=4.0,
        num_layers=2,
        num_average_neigh=1.0,
        output_signature={0: 2, 2: 2, 4: 1},
    )

    # R f(x)
    N = num_atoms.view(1)  # make it a 1D tensor
    nat_tensor = model(edge_vector, edge_idx, atom_type, N)
    fx = converter.to_ordinary_tensor(nat_tensor)
    fx = fx.squeeze(0)  # remove batch dimension
    Rfx = torch.einsum("ijkl,ai,bj,ck,dl->abcd", fx, R, R, R, R)

    # Rotated elastic tensor f(Rx)
    rotated_nat_tensor = model(rotated_edge_vector, edge_idx, atom_type, N)
    fRx = converter.to_ordinary_tensor(rotated_nat_tensor)
    fRx = fRx.squeeze(0)  # remove batch dimension

    assert torch.allclose(Rfx, fRx, rtol=1e-5, atol=1e-6)


def test_inversion(config_info):
    """
    Test inversion of the StructureTensorModel.
    f(x) = f(-x) for even parity
    f(x) = -f(-x) for odd parity

    """
    seed_everything(35)

    _, atom_type, edge_vector, edge_idx, num_atoms, num_atom_types = config_info

    # tensor
    signature = {0: 1, 1: 1, 2: 1, 3: 1, 4: 1}
    model = StructureTensorModel(
        F=6,
        max_L=4,
        num_atom_types=num_atom_types,
        r_cut=4.0,
        num_layers=2,
        num_average_neigh=1.0,
        output_signature=signature,
        tp_path_mode="full",
        tp_path_polar_only=True,  # This should be `True` to ensure inversion sym
    )

    # f(x)
    N = num_atoms.view(1)  # make it a 1D tensor
    nat_tensor = model(edge_vector, edge_idx, atom_type, N)

    # f(-x)
    inverted_edge_vector = -edge_vector
    inverted_nat_tensor = model(inverted_edge_vector, edge_idx, atom_type, N)

    for l, n in signature.items():
        v1 = nat_tensor[l]
        v2 = inverted_nat_tensor[l]
        if l % 2 == 0:  # even rank polar tensor, sign not change
            assert torch.allclose(v1, v2, rtol=1e-5, atol=1e-6)
        else:  # odd rank polar tensor, sign changes
            assert torch.allclose(v1, -v2, rtol=1e-5, atol=1e-6)
