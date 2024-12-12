from carten.core.utils import is_symmetric, is_traceless
from carten.module.hyper_moment import HyperMoment

from ..conftest import create_feature_tensors


def test_HyperMoment(config_info):
    coords, atom_type, edge_vector, edge_idx, num_atoms, num_atom_types = config_info

    F = 5
    L = 4
    max_out_L = 4
    x = create_feature_tensors(num_atoms, F, L)

    hm = HyperMoment(F, L, max_out_L)
    out = hm(x)

    # Check the total shape
    assert out.shape == (num_atoms, F, (3 ** (max_out_L + 1) - 1) // 2)

    # Check each l is a natural tensor
    for l3 in range(max_out_L + 1):
        sliced = out[..., (3**l3 - 1) // 2 : (3 ** (l3 + 1) - 1) // 2]

        # change the shape to (num_atoms, F, 3, 3, ...)
        sliced = sliced.reshape(num_atoms, F, *(3,) * l3)

        assert is_symmetric(sliced, start_dim=2, atol=1e-6), f"l={l3} is not symmetric"
        assert is_traceless(sliced, start_dim=2, atol=1e-6), f"l={l3} is not traceless"
