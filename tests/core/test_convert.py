import pytest
import torch
from natt.convert import Converter
from natt.sym import symmetrize


@pytest.mark.parametrize(
    "rank,symmetry",
    [
        (2, None),
        (2, "ij=ji"),
        (3, None),
        (3, "ijk=ikj"),
        (3, "ijk=ikj=jik"),
        (4, None),
        (4, "ijkl=jikl=klij"),
        (4, "ijkl=jikl=kjil=ljki"),
    ],
)
def test_Converter(rank, symmetry):

    torch.manual_seed(35)
    T = torch.randn(*[3] * rank)

    converter = Converter(rank, symmetry)

    # symmetrize the tensor if `symmetry` is not None
    if symmetry is not None:
        T = symmetrize(T, symmetry)

    X = converter.to_natural_tensor(T)
    T_2 = converter.to_ordinary_tensor(X)

    assert torch.allclose(T, T_2, atol=1e-6), f"Failed for rank {rank}."
