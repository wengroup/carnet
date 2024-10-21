import torch

from carten.natural_tensor import NaturalTensors
from carten.reduce import (
    get_dyadic_tensor,
    get_permutations,
    reduce_symmetric_tensor,
    symmetrize,
)
from carten.utils import check_symmetric, check_traceless


def test_reduce_symmetric_tensor(T3, T4):
    t3 = symmetrize(T3)
    t4 = symmetrize(T4)

    t3_out = reduce_symmetric_tensor(t3)
    assert isinstance(t3_out, NaturalTensors)
    assert t3_out.signature == [(1, 3), (1, 1)]
    for t in t3_out:
        assert check_symmetric(t, atol=1e-4)
        assert check_traceless(t, atol=1e-4)

    t4_out = reduce_symmetric_tensor(t4)
    assert isinstance(t4_out, NaturalTensors)
    assert t4_out.signature == [(1, 4), (1, 2), (1, 0)]
    for t in t4_out:
        assert check_symmetric(t, atol=1e-4)
        assert check_traceless(t, atol=1e-4)

    # batched
    t3_b = torch.vstack([t3, t3]).reshape(2, *t3.shape)
    t3_b_out = reduce_symmetric_tensor(t3_b, start_dim=1)
    assert isinstance(t3_b_out, NaturalTensors)
    assert t3_b_out.signature == [(1, 3), (1, 1)]
    for t in t3_b_out:
        assert check_symmetric(t, atol=1e-4, start_dim=1)
        assert check_traceless(t, atol=1e-4, start_dim=1)


def test_get_dyadic_tensor():
    r = torch.tensor([1.0, 2.0, 3.0])

    t = get_dyadic_tensor(r, rank=3, normalize=False)
    ref = torch.einsum("i,j,k->ijk", r, r, r)
    assert torch.allclose(t, ref)

    # batched
    r2 = torch.vstack([r, r])
    t2 = get_dyadic_tensor(r2, rank=3, normalize=False)
    ref2 = torch.cat([ref, ref]).reshape(2, *ref.shape)
    assert torch.allclose(t2, ref2)

    t = get_dyadic_tensor(r, rank=3, normalize=True)
    nr = r / torch.norm(r)
    ref = torch.einsum("i,j,k->ijk", nr, nr, nr)
    assert torch.allclose(t, ref)


def test_get_permutations():
    ref = [
        [0, 1, 2, 3, 4],
        [0, 1, 3, 2, 4],
        [0, 1, 3, 4, 2],
        [0, 3, 1, 2, 4],
        [0, 3, 1, 4, 2],
        [0, 3, 4, 1, 2],
        [3, 0, 1, 2, 4],
        [3, 0, 1, 4, 2],
        [3, 0, 4, 1, 2],
        [3, 4, 0, 1, 2],
    ]
    perms = get_permutations("aaabb")
    assert perms == ref

    perms = get_permutations("aaabb", start_dim=2)
    assert perms == [[0, 1] + [2 + i for i in sub] for sub in ref]


def test_symmetrize():
    t = torch.arange(81).reshape(3, 3, 3, 3).to(torch.float32)

    u = symmetrize(t)
    check_symmetric(u)

    u = symmetrize(t, start_dim=2)
    check_symmetric(u, start_dim=2)
