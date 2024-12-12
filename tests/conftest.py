import pytest
import torch

from carten.core.reduce import symmetrize_and_remove_trace
from carten.data.utils import get_edge_vec


@pytest.fixture(scope="session")
def T0():
    return get_T(0)


@pytest.fixture(scope="session")
def T1():
    return get_T(1)


@pytest.fixture(scope="session")
def T2():
    return get_T(2)


@pytest.fixture(scope="session")
def T3():
    return get_T(3)


@pytest.fixture(scope="session")
def T4():
    return get_T(4)


@pytest.fixture(scope="session")
def NT0():
    return get_NT(0)


@pytest.fixture(scope="session")
def NT1():
    return get_NT(1)


@pytest.fixture(scope="session")
def NT2():
    return get_NT(2)


@pytest.fixture(scope="session")
def NT3():
    return get_NT(3)


@pytest.fixture(scope="session")
def NT4(T4):
    return get_NT(4)


@pytest.fixture(scope="session")
def T0_chunk(NT0, mul=2):
    # adding a dummy leading dimension 1
    return torch.stack([NT0, NT0]).reshape(1, mul * 1)


@pytest.fixture(scope="session")
def T0_shaped_chunk(NT0, mul=2):
    # adding a dummy leading dimension 1
    return torch.stack([NT0, NT0]).reshape(1, mul, 1)


@pytest.fixture(scope="session")
def T1_chunk(NT1, mul=2):
    return torch.cat([NT1.flatten()] * mul, dim=-1).reshape(1, mul * 3)


@pytest.fixture(scope="session")
def T1_shaped_chunk(NT1, mul=2):
    return torch.cat([NT1.flatten()] * mul, dim=-1).reshape(1, mul, 3)


@pytest.fixture(scope="session")
def T2_chunk(NT2, mul=2):
    return torch.cat([NT2.flatten()] * mul, dim=-1).reshape(1, mul * 9)


@pytest.fixture(scope="session")
def T2_shaped_chunk(NT2, mul=2):
    return torch.cat([NT2.flatten()] * mul, dim=-1).reshape(1, mul, 3, 3)


@pytest.fixture(scope="session")
def T3_chunk(NT3, mul=2):
    return torch.cat([NT3.flatten()] * mul, dim=-1).reshape(1, mul * 27)


@pytest.fixture(scope="session")
def T3_shaped_chunk(NT3, mul=2):
    return torch.cat([NT3.flatten()] * mul, dim=-1).reshape(1, mul, 3, 3, 3)


@pytest.fixture(scope="session")
def T4_chunk(NT4, mul=2):
    return torch.cat([NT4.flatten()] * mul, dim=-1).reshape(1, mul * 81)


@pytest.fixture(scope="session")
def T4_shaped_chunk(T4, mul=2):
    return torch.cat([T4.flatten()] * mul, dim=-1).reshape(1, mul, 3, 3, 3, 3)


def get_T(rank: int):
    """Create a tensor of rank `rank` for testing."""
    if rank == 0:
        return torch.tensor(1.0)
    t = torch.arange(3**rank).reshape([3] * rank).to(torch.float32)
    return t / t.mean()


def get_NT(rank: int):
    """Create a natural tensor of rank `rank` for testing."""
    t = get_T(rank)
    return symmetrize_and_remove_trace(t)


@pytest.fixture(scope="session")
def config_info():
    """Create an atomic configuration with 4 atoms and 2 atom types."""
    # 4 atoms
    atom_types = torch.tensor([0, 1, 1, 1])
    coords = 0.2 * torch.arange(12).reshape(4, 3).to(torch.get_default_dtype())
    coords[0, 0] += 0.1
    coords[1, 1] += 0.2
    coords[2, 2] += 0.3
    coords[3, 1] += 0.1

    coords.requires_grad_(True)

    edge_idx = torch.tensor(
        [
            [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3],
            [1, 2, 3, 0, 2, 3, 0, 1, 3, 0, 1, 2],
        ]
    )
    edge_vector = torch.stack([coords[j] - coords[i] for i, j in edge_idx.T])

    num_atoms = torch.max(edge_idx) + 1
    num_atom_types = atom_types.max() + 1

    return coords, atom_types, edge_vector, edge_idx, num_atoms, num_atom_types


@pytest.fixture(scope="session")
def batched_config_info(config_info):
    """Create a batched of two atomic configurations."""
    coords, atom_types, edge_vector, edge_idx, num_atoms, num_atom_types = config_info

    coords = torch.vstack([coords, coords])
    atom_types = torch.hstack([atom_types, atom_types])
    edge_idx = torch.hstack([edge_idx, edge_idx + num_atoms])
    num_atoms = torch.tensor([num_atoms, num_atoms])
    num_atom_types = num_atom_types

    shift_vec = torch.zeros_like(edge_vector)
    shift_vec = torch.vstack([shift_vec, shift_vec])
    cell = torch.eye(3)
    cell = torch.vstack([cell, cell])
    batch = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])

    edge_vector = get_edge_vec(coords, shift_vec, cell, edge_idx, batch=batch)

    return coords, atom_types, edge_vector, edge_idx, num_atoms, num_atom_types


def create_feature_tensors(n_atoms: int, F: int, L: int):
    """
    Create atomic features for testing.

    Args:
        n_atoms: number of atoms
        F: channel dimension
        L: maximum rank of natural tensors, the feature tensor will have ranks up to L.

    Returns:
        Tensor of shape (n_atoms, F, T), where T = (3**(L+1) - 1) // 2. The values
        across n_atoms and F dims are the same.

    """
    x = torch.cat([get_NT(l).view(-1) for l in range(L + 1)])
    x = x.repeat(n_atoms, F, 1)

    return x
