import torch


def test_get_atomic_number(dataset):
    assert dataset.get_atomic_number() == [6, 14]

    assert torch.allclose(dataset.get_mean_atomic_energy(), torch.tensor(-4.2591))

    assert torch.allclose(
        dataset.get_root_mean_square_force(), torch.tensor(0.6932), atol=1e-4
    )

    assert torch.allclose(dataset.get_num_average_neigh(), torch.tensor(28.0))
