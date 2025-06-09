import pytest
import torch
from natt.sym import symmetrize

from carten.core.convert import Converter


@pytest.mark.parametrize(
    "symmetry",
    [
        "ij",
        "ij=ji",
        "ijk",
        "ijk=ikj",
        "ijk=ikj=jik",
        "ijkl",
        "ijkl=jikl=klij",
        "ijkl=jikl=kjil=ljki",
    ],
)
def test_Converter(symmetry):

    torch.manual_seed(35)

    rank = len(set(symmetry.replace("=", "").replace(" ", "")))
    T = torch.randn(*[3] * rank)

    converter = Converter(symmetry)

    # symmetrize the tensor if `symmetry` is not None
    if symmetry is not None:
        T = symmetrize(T, symmetry)

    X = converter.to_natural_tensor(T)
    T_2 = converter.to_ordinary_tensor(X)

    assert torch.allclose(T, T_2, atol=1e-6), f"Failed for rank {rank}."
