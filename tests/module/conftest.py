import pytest
import torch

from carten.core.reduce import symmetrize_and_remove_trace
from carten.data.utils import get_edge_vec


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


def get_NT(rank: int):
    """Create a natural tensor of rank `rank` for testing."""
    if rank == 0:
        return torch.tensor(1.0)

    t = torch.arange(3**rank).reshape([3] * rank).to(torch.float32)
    t = t / t.mean()

    return symmetrize_and_remove_trace(t)
