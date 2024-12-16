import torch

from carten.module.scatter import scatter


def test_scatter():
    x = torch.tensor([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]])
    index = torch.tensor([0, 1, 0, 1, 0])

    out = scatter(x, index, reduce="sum")
    assert torch.allclose(out, torch.tensor([[6, 4], [21, 14]]))

    out = scatter(x, index, reduce="mean")
    assert torch.allclose(out, torch.tensor([[2, 2], [7, 7]]))

    out = scatter(x, index, reduce="max")
    assert torch.allclose(out, torch.tensor([[4, 3], [9, 8]]))

    out = scatter(x, index, reduce="min")
    assert torch.allclose(out, torch.tensor([[0, 1], [5, 6]]))
