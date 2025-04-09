import pytest
import torch

from carten.symbolic.convert import Converter


@pytest.mark.parametrize("rank", [2, 3, 4])
def test_Converter(rank):

    torch.manual_seed(35)

    converter = Converter(rank)

    T = torch.randn(*[3] * rank)

    X = converter.to_natural_tensor(T)
    T_2 = converter.to_ordinary_tensor(X)

    assert torch.allclose(T, T_2, atol=1e-6), f"Failed for rank {rank}."
