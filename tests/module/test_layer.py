from natt.utils import is_symmetric, is_traceless
import torch

from carnet.module.layer import Layer
from carnet.module.radial import RadialBasis

from ..conftest import create_feature_tensors


def test_Layer(config_info):
    coords, atom_type, edge_vector, edge_idx, num_atoms, num_atom_types = config_info

    F = 5
    L1 = L2 = L3 = 4
    max_out_L = 4

    x = create_feature_tensors(num_atoms, F, L1)

    # Need radial basis
    radial_basis_degree = 8
    r_cut = 5.0
    radial = RadialBasis(radial_basis_degree, r_cut, envelope=6)

    radial_basis = radial(
        torch.linalg.vector_norm(edge_vector, dim=-1),
    )

    layer = Layer(
        F,
        L1,
        L2,
        L3,
        num_atom_types=num_atom_types,
        num_average_neigh=1.0,
        radial_output_dim=radial.output_dim,
        max_out_L=max_out_L,
    )

    out = layer(edge_vector, edge_idx, atom_type, x, radial_basis, None)

    # Check the total shape
    assert out.shape == (num_atoms, F, (3 ** (max_out_L + 1) - 1) // 2)

    # Check each l is a natural tensor
    for l3 in range(max_out_L + 1):
        sliced = out[..., (3**l3 - 1) // 2 : (3 ** (l3 + 1) - 1) // 2]

        # change the shape to (num_atoms, F, 3, 3, ...)
        sliced = sliced.reshape(num_atoms, F, *(3,) * l3)

        assert is_symmetric(sliced, start_dim=2, atol=1e-6), f"l={l3} is not symmetric"
        assert is_traceless(sliced, start_dim=2, atol=1e-5), f"l={l3} is not traceless"
