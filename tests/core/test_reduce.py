import torch

from carten.natural_tensor import NaturalTensors
from carten.core.reduce import (
    get_contraction_with_delta_rules,
    get_dyadic_tensor, reduce_symmetric_tensor,
    remove_trace,
    remove_trace_rule,
    symmetrize,
)
from carten.core.permute import get_permutations, get_permutations_2
from carten.core.utils import is_symmetric, is_traceless


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
    assert get_permutations("aaaa") == [[0, 1, 2, 3]]
    assert get_permutations("aaaa", start_dim=2) == [[0, 1, 2, 3, 4, 5]]

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


def test_get_permutations_2():
    perms = get_permutations_2(m=2, num_delta=1)
    assert perms == [[0, 1]]


def test_symmetrize(T2, T3, T4):
    for t in [T2, T3, T4]:
        for start_dim in range(2):
            sym = symmetrize(t, start_dim)
            is_symmetric(sym, start_dim)


def test_remove_trace_rule():
    rule = remove_trace_rule(5, 2)
    assert rule == "...aabbc,de,fg->...cdefg"


def test_remove_trace(T2, T3, T4):
    """
    Test traceless part, not the symmetric part.
    """
    # second rank tensor
    t2 = symmetrize(T2)
    t2_1 = remove_trace(t2, start_dim=0)
    assert torch.einsum("ii", t2_1) == 0.0

    t2_2 = t2.reshape(1, 1, 3, 3)
    t2_tl = remove_trace(t2_2, start_dim=2)
    assert t2_tl.shape == t2_2.shape

    assert torch.einsum("...ii", t2_tl) == 0.0

    # third rank tensor
    t3 = symmetrize(T3)
    t3 = t3.reshape(1, 1, 3, 3, 3)
    t3_tl = remove_trace(t3, start_dim=2)
    assert t3_tl.shape == t3.shape

    for rule in ["...iij", "...iji", "...jii"]:
        out = torch.einsum(rule, t3_tl)
        assert torch.allclose(out, torch.zeros(3), atol=1e-5)

    # fourth rank tensor
    t4 = symmetrize(T4)
    t4 = t4.reshape(1, 1, 3, 3, 3, 3)
    t4_tl = remove_trace(t4, start_dim=2)
    assert t4_tl.shape == t4.shape

    for rule in ["...iijk", "...ijik", "...ijki", "...jiik", "...jiki", "...jkii"]:
        out = torch.einsum(rule, t4_tl)
        assert torch.allclose(out, torch.zeros(3, 3), atol=1e-4)


def test_get_contraction_with_delta_rules():
    out = get_contraction_with_delta_rules(3, 1)
    assert out == ["abc,bc->a", "abc,ac->b", "abc,ab->c"]

    out = get_contraction_with_delta_rules(4, 1)
    assert set(out) == {
        "abcd,cd->ab",
        "abcd,bd->ac",
        "abcd,bc->ad",
        "abcd,ad->bc",
        "abcd,ac->bd",
        "abcd,ab->cd",
    }

    out = get_contraction_with_delta_rules(4, 2)
    assert set(out) == {
        "abcd,ab,cd->",
        "abcd,ac,bd->",
        "abcd,ad,bc->",
    }

    out = get_contraction_with_delta_rules(5, 1)
    assert set(out) == {
        "abcde,ab->cde",
        "abcde,ac->bde",
        "abcde,ad->bce",
        "abcde,ae->bcd",
        #
        "abcde,bc->ade",
        "abcde,bd->ace",
        "abcde,be->acd",
        #
        "abcde,cd->abe",
        "abcde,ce->abd",
        #
        "abcde,de->abc",
    }

    out = get_contraction_with_delta_rules(5, 2)
    assert set(out) == {
        "abcde,bc,de->a",
        "abcde,bd,ce->a",
        "abcde,be,cd->a",
        #
        "abcde,ac,de->b",
        "abcde,ad,ce->b",
        "abcde,ae,cd->b",
        #
        "abcde,ab,de->c",
        "abcde,ad,be->c",
        "abcde,ae,bd->c",
        #
        "abcde,ab,ce->d",
        "abcde,ac,be->d",
        "abcde,ae,bc->d",
        #
        "abcde,ab,cd->e",
        "abcde,ac,bd->e",
        "abcde,ad,bc->e",
    }
