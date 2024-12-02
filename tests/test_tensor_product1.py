import torch

from carten.natural_tensor import NaturalTensors
from carten.reduce import symmetrize, symmetrize_and_remove_trace
from carten.tensor_product1 import (
    TensorProduct,
    get_asym_part,
    get_delta_contraction_rule,
    get_epsilon_delta_contraction_rule,
    get_sym_part,
    get_sym_rules_2,
    get_sym_rules_3,
    tp,
)
from carten.utils import is_symmetric, is_traceless, eijk


def test_get_delta_contraction_rule():
    rule = get_delta_contraction_rule("abc", "def", num_delta=0)
    assert rule == "...abc, ...def"

    rule = get_delta_contraction_rule("abc", "def", num_delta=1)
    assert rule == "be, ...abc, ...def"

    rule = get_delta_contraction_rule("abc", "def", num_delta=2)
    assert rule == "be, cf, ...abc, ...def"


def test_get_epsilon_delta_contraction_rule():
    rule = get_epsilon_delta_contraction_rule("abc", "def", num_delta=0)
    assert rule == "zbe, ...abc, ...def"

    rule = get_epsilon_delta_contraction_rule("abc", "def", num_delta=1)
    assert rule == "zcf, be, ...abc, ...def"


def test_get_sym_rules_2():
    rules = get_sym_rules_2("a", "b")
    assert set(rules) == {"ab", "ba"}

    rules = get_sym_rules_2("a", "cd")
    assert set(rules) == {"acd", "cad", "cda"}

    rules = get_sym_rules_2("ab", "cd")
    assert set(rules) == {"abcd", "acbd", "acdb", "cabd", "cadb", "cdab"}

    rules = get_sym_rules_2("ab", "def")
    assert set(rules) == {
        "abdef",
        "adbef",
        "adebf",
        "adefb",
        #
        "dabef",
        "daebf",
        "daefb",
        #
        "deabf",
        "deafb",
        #
        "defab",
    }


def test_get_sym_rules_3():
    rules = get_sym_rules_3("a", "b", "c")
    assert set(rules) == {"abc", "acb", "bac", "bca", "cab", "cba"}

    rules = get_sym_rules_3("a", "bc", "de")
    assert set(rules) == {
        "abcde",
        "abdce",
        "abdec",
        "adbce",
        "adbec",
        "adebc",
        #
        "bacde",
        "badce",
        "badec",
        "dabce",
        "dabec",
        "daebc",
        #
        "bcade",
        "bdace",
        "bdaec",
        "dbace",
        "dbaec",
        "deabc",
        #
        "bcdae",
        "bdcae",
        "bdeac",
        "dbcae",
        "dbeac",
        "debac",
        #
        "bcdea",
        "bdcea",
        "bdeca",
        "dbcea",
        "dbeca",
        "debca",
    }


def test_get_sym_part(T3, T4):
    S = symmetrize(T3 + 0.5)
    T = symmetrize(T4 + 0.7)

    rank_S = S.ndim
    rank_T = T.ndim

    # add dummy batching and multiplicity directions
    S = S.reshape(1, 1, *S.shape)
    T = T.reshape(1, 1, *T.shape)

    # added dim for result, 1 for batch and 1 for multiplicity
    added_dim = 2

    U = get_sym_part(S, T, rank_S, rank_T, num_delta=0)
    assert U.ndim == rank_S + rank_T + added_dim
    assert is_symmetric(U, start_dim=added_dim)

    U = get_sym_part(S, T, rank_S, rank_T, num_delta=1)
    assert U.ndim == rank_S + rank_T + added_dim - 2
    assert is_symmetric(U, start_dim=added_dim)

    U = get_sym_part(S, T, rank_S, rank_T, num_delta=2)
    assert U.ndim == rank_S + rank_T + added_dim - 4
    assert is_symmetric(U, start_dim=added_dim)

    U = get_sym_part(S, T, rank_S, rank_T, num_delta=3)
    assert U.ndim == rank_S + rank_T + added_dim - 6
    assert is_symmetric(U, start_dim=added_dim)


def test_get_asym_part(T3, T4):
    S = symmetrize_and_remove_trace(T4 + 0.5)
    T = symmetrize_and_remove_trace(T3 + 0.7)

    rank_S = S.ndim
    rank_T = T.ndim

    # add dummy batching and multiplicity directions
    S = S.reshape(1, 1, *S.shape)
    T = T.reshape(1, 1, *T.shape)

    # added dim for result, 1 for batch and 2 for multiplicity dimensions
    added_dim = 2

    U = get_asym_part(S, T, rank_S, rank_T, num_delta=0)
    assert U.ndim == rank_S + rank_T + added_dim - 1
    assert is_symmetric(U, start_dim=added_dim, atol=1e-4)

    U = get_asym_part(S, T, rank_S, rank_T, num_delta=1)
    assert U.ndim == rank_S + rank_T + added_dim - 3
    assert is_symmetric(U, start_dim=added_dim, atol=1e-4)

    U = get_asym_part(S, T, rank_S, rank_T, num_delta=2)
    assert U.ndim == rank_S + rank_T + added_dim - 5
    assert is_symmetric(U, start_dim=added_dim, atol=1e-4)


def test_tp(T2, T4):
    """Test the generated irreps are symmetric and traceless."""
    S = symmetrize_and_remove_trace(T4 + 0.5)
    T = symmetrize_and_remove_trace(T2 + 0.7)

    rank_S = 4
    rank_T = 2

    # add dummy batching and multiplicity directions
    S = S.reshape(1, 1, *S.shape)
    T = T.reshape(1, 1, *T.shape)

    added_dim = 2

    irreps = tp(S, T, rank_S, rank_T)

    assert len(irreps) == 5

    for i, t in enumerate(irreps):
        assert t.ndim == added_dim + rank_S - rank_T + i
        assert is_symmetric(t, start_dim=added_dim, atol=1e-4)
        assert is_traceless(t, start_dim=added_dim, atol=1e-4)

    irreps = tp(S, T, rank_S=4, rank_T=2, out_ranks=[3, 4, 5])
    assert len(irreps) == 3
    for i, t in enumerate(irreps):
        assert t.ndim == added_dim + 3 + i
        assert is_symmetric(t, start_dim=added_dim, atol=1e-4)
        assert is_traceless(t, start_dim=added_dim, atol=1e-4)


def test_tp_2():
    """Test the TP of two rank-1 tensor decomposes like a 2-rank tensor,
    giving the correct trace, antisymmetric and symmetric parts.
    """
    S = torch.arange(1, 4).to(torch.float32)
    T = torch.arange(2, 5).to(torch.float32)

    # reference
    ST = torch.einsum("a,b->ab", S, T)
    trace = ST.trace()
    asym = (ST - ST.T) / 2
    asym = torch.einsum("zab,ab->z", eijk(), asym)
    sym = (ST + ST.T) / 2 - trace * torch.eye(3) / 3

    irreps = tp(S.reshape(1, 1, 3), T.reshape(1, 1, 3), rank_S=1, rank_T=1)

    assert len(irreps) == 3

    # the later [0][0] is to remove the batch and multiplicity dimensions
    assert torch.allclose(irreps[0][0][0], trace)
    assert torch.allclose(irreps[1][0][0], asym)
    assert torch.allclose(irreps[2][0][0], sym)


def test_TensorProduct(T2, T3):
    S = symmetrize_and_remove_trace(T2)
    T = symmetrize_and_remove_trace(T3)

    nt1 = nt2 = NaturalTensors.from_sequence([S, T])
    tp = TensorProduct()
    out = tp(nt1, nt2)

    out = out.regroup()

    assert out.signature == [(2, 0), (4, 1), (4, 2), (4, 3), (4, 4), (3, 5), (1, 6)]
