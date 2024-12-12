from carten.core.utils import is_symmetric, is_traceless
from carten.module.atomic_moment import AtomicMoment

from ..conftest import create_feature_tensors


def test_AtomicMoment(config_info):
    coords, atom_type, edge_vector, edge_idx, num_atoms, num_atom_types = config_info

    F = 5
    L1 = L2 = L3 = 4
    x = create_feature_tensors(num_atoms, F, L1)

    am = AtomicMoment(F, L1, L2, L3, num_atom_types, num_average_neigh=1.0)
    out = am(edge_vector, edge_idx, atom_type, x)

    # Check the total shape
    assert out.shape == (num_atoms, F, (3 ** (L3 + 1) - 1) // 2)

    # Check each l is a natural tensor
    for l3 in range(L3 + 1):
        sliced = out[..., (3**l3 - 1) // 2 : (3 ** (l3 + 1) - 1) // 2]

        # change the shape to (num_atoms, F, 3, 3, ...)
        sliced = sliced.reshape(num_atoms, F, *(3,) * l3)

        assert is_symmetric(sliced, start_dim=2, atol=1e-6), f"l={l3} is not symmetric"
        assert is_traceless(sliced, start_dim=2, atol=1e-6), f"l={l3} is not traceless"
