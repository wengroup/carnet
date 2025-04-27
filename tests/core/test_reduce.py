import torch
from natt.symmetrize import symmetrize
from natt.utils import is_symmetric, is_traceless

from carten.core.reduce import reduce_symmetric_tensor
from carten.natural_tensor import NaturalTensors


def test_reduce_symmetric_tensor(T3, T4):
    t3 = symmetrize(T3)
    t4 = symmetrize(T4)

    t3_out = reduce_symmetric_tensor(t3)
    assert isinstance(t3_out, NaturalTensors)
    assert t3_out.signature == [(1, 3), (1, 1)]
    for t in t3_out:
        assert is_symmetric(t, atol=1e-4)
        assert is_traceless(t, atol=1e-4)

    t4_out = reduce_symmetric_tensor(t4)
    assert isinstance(t4_out, NaturalTensors)
    assert t4_out.signature == [(1, 4), (1, 2), (1, 0)]
    for t in t4_out:
        assert is_symmetric(t, atol=1e-4)
        assert is_traceless(t, atol=1e-4)

    # batched
    t3_b = torch.vstack([t3, t3]).reshape(2, *t3.shape)
    t3_b_out = reduce_symmetric_tensor(t3_b, start_dim=1)
    assert isinstance(t3_b_out, NaturalTensors)
    assert t3_b_out.signature == [(1, 3), (1, 1)]
    for t in t3_b_out:
        assert is_symmetric(t, atol=1e-4, start_dim=1)
        assert is_traceless(t, atol=1e-4, start_dim=1)
