import torch

from carnet.module.scatter import scatter


def test_scatter():
    # src: (2, 5) -> We want to scatter on dim 1 (size 5) to say size 2
    # But AtomicMoment scatters on dim 0.
    # Let's use src: (5, 2), index: (5,)
    x = torch.tensor([[0.0, 5.0], [1.0, 6.0], [2.0, 7.0], [3.0, 8.0], [4.0, 9.0]])
    index = torch.tensor([0, 1, 0, 1, 0])

    # sum
    out = scatter(x, index, reduce="sum", dim=0)
    # 0: [0,5] + [2,7] + [4,9] = [6, 21]
    # 1: [1,6] + [3,8] = [4, 14]
    expected_sum = torch.tensor([[6.0, 21.0], [4.0, 14.0]])
    assert torch.allclose(out, expected_sum)

    # mean
    out = scatter(x, index, reduce="mean", dim=0)
    # 0: [6/3, 21/3] = [2, 7]
    # 1: [4/2, 14/2] = [2, 7]
    expected_mean = torch.tensor([[2.0, 7.0], [2.0, 7.0]])
    assert torch.allclose(out, expected_mean)

    # max
    out = scatter(x, index, reduce="amax", dim=0)
    # 0: max([0,5], [2,7], [4,9]) = [4, 9]
    # 1: max([1,6], [3,8]) = [3, 8]
    expected_max = torch.tensor([[4.0, 9.0], [3.0, 8.0]])
    assert torch.allclose(out, expected_max)

    # min
    out = scatter(x, index, reduce="amin", dim=0)
    # 0: min([0,5], [2,7], [4,9]) = [0, 5]
    # 1: min([1,6], [3,8]) = [1, 6]
    expected_min = torch.tensor([[0.0, 5.0], [1.0, 6.0]])
    assert torch.allclose(out, expected_min)


def test_scatter_3d():
    # src: (5, 2, 2), index: (5,)
    x = torch.ones(5, 2, 2)
    index = torch.tensor([0, 0, 1, 1, 1])

    out = scatter(x, index, reduce="sum", dim=0)
    assert out.shape == (2, 2, 2)
    assert torch.allclose(out[0], torch.full((2, 2), 2.0))
    assert torch.allclose(out[1], torch.full((2, 2), 3.0))
