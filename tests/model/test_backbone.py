from natt.utils import is_symmetric, is_traceless

from carnet.model.backbone import Backbone


def test_Backbone(batched_config_info):
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
    max_out_L = 3
    num_layers = 2

    module = Backbone(
        F=F,
        max_L=max_L,
        num_atom_types=num_atom_types,
        r_cut=5.0,
        num_layers=num_layers,
        num_average_neigh=1.0,
        max_out_L=max_out_L,
    )

    out = module(
        edge_vector, edge_idx, atom_type, num_atoms, return_all=False, scalar_only=False
    )

    # Check the total shape
    out = out[0]
    assert out.shape == (total_num_atoms, F, (3 ** (max_out_L + 1) - 1) // 2)

    # Check each l is a natural tensor
    for l3 in range(max_out_L + 1):
        sliced = out[..., (3**l3 - 1) // 2 : (3 ** (l3 + 1) - 1) // 2]

        # change the shape to (num_atoms, F, 3, 3, ...)
        sliced = sliced.reshape(total_num_atoms, F, *(3,) * l3)

        assert is_symmetric(sliced, start_dim=2, atol=1e-6), f"l={l3} is not symmetric"
        assert is_traceless(sliced, start_dim=2, atol=1e-5), f"l={l3} is not traceless"

    # Check other return modes
    out = module(
        edge_vector, edge_idx, atom_type, num_atoms, return_all=False, scalar_only=True
    )
    out = out[0]
    assert out.shape == (total_num_atoms, F, 1)

    #
    out = module(
        edge_vector, edge_idx, atom_type, num_atoms, return_all=True, scalar_only=False
    )
    assert len(out) == num_layers
    out = out[0]
    assert out.shape == (total_num_atoms, F, (3 ** (max_out_L + 1) - 1) // 2)

    #
    out = module(
        edge_vector, edge_idx, atom_type, num_atoms, return_all=True, scalar_only=True
    )
    assert len(out) == num_layers
    out = out[0]
    assert out.shape == (total_num_atoms, F, 1)
