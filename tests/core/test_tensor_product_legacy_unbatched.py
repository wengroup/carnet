import torch

from carten.core.reduce import symmetrize, symmetrize_and_remove_trace
from carten.core.tensor_product_legacy_unbatched import (
    get_asym_part,
    get_delta_contraction_rule,
    get_epsilon_delta_contraction_rule,
    get_sym_part,
    get_sym_rules_2,
    get_sym_rules_3,
    tp,
)
from carten.core.utils import eijk, is_symmetric, is_traceless


def test_get_delta_contraction_rule():
    rule = get_delta_contraction_rule("abc", "def", num_delta=0)
    assert rule == "abc,def"

    rule = get_delta_contraction_rule("abc", "def", num_delta=1)
    assert rule == "ad,abc,def"

    rule = get_delta_contraction_rule("abc", "def", num_delta=2)
    assert rule == "ad,be,abc,def"

    rule = get_delta_contraction_rule("abc", "def", num_delta=3)
    assert rule == "ad,be,cf,abc,def"


def test_get_epsilon_delta_contraction_rule():
    rule = get_epsilon_delta_contraction_rule("abc", "def", num_delta=0)
    assert rule == "zad,abc,def"

    rule = get_epsilon_delta_contraction_rule("abc", "def", num_delta=1)
    assert rule == "zbe,ad,abc,def"

    rule = get_epsilon_delta_contraction_rule("abc", "def", num_delta=2)
    assert rule == "zcf,ad,be,abc,def"


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

    U = get_sym_part(S, T, num_delta=0)
    assert U.ndim == S.ndim + T.ndim
    assert is_symmetric(U)

    U = get_sym_part(S, T, num_delta=1)
    assert U.ndim == S.ndim + T.ndim - 2
    assert is_symmetric(U)

    U = get_sym_part(S, T, num_delta=2)
    assert U.ndim == S.ndim + T.ndim - 4
    assert is_symmetric(U)


def test_get_asym_part(T3, T4):
    S = symmetrize_and_remove_trace(T4 + 0.5)
    T = symmetrize_and_remove_trace(T3 + 0.7)

    U = get_asym_part(S, T, num_delta=0)
    assert U.ndim == S.ndim + T.ndim - 1
    assert is_symmetric(U, atol=1e-4)

    U = get_asym_part(S, T, num_delta=1)
    assert U.ndim == S.ndim + T.ndim - 3
    assert is_symmetric(U, atol=1e-4)

    U = get_asym_part(S, T, num_delta=2)
    assert U.ndim == S.ndim + T.ndim - 5
    assert is_symmetric(U, atol=1e-4)


def test_tp(T2, T4):
    """Test the generated irreps are symmetric and traceless."""
    S = symmetrize_and_remove_trace(T4 + 0.5)
    T = symmetrize_and_remove_trace(T2 + 0.7)

    irreps = tp(S, T)

    assert len(irreps) == 5

    for i, t in enumerate(irreps):
        assert t.ndim == abs(S.ndim - T.ndim) + i
        assert is_symmetric(t, atol=1e-4)
        assert is_traceless(t, atol=1e-4)

    irreps = tp(S, T, min_rank=3, max_rank=5)
    assert len(irreps) == 3
    for i, t in enumerate(irreps):
        assert t.ndim == 3 + i
        assert is_symmetric(t, atol=1e-4)
        assert is_traceless(t, atol=1e-4)


def test_tp_2():
    """Test the TP of two rank-1 tensor decomposes like a 2-rank tensor,
    giving the correct trace, antisymmetric and symmetric parts.
    """
    S = torch.arange(1, 4).to(torch.float32)
    T = torch.arange(2, 5).to(torch.float32)

    ST = torch.einsum("a,b->ab", S, T)
    trace = ST.trace()
    asym = (ST - ST.T) / 2
    asym = torch.einsum("zab,ab->z", eijk(), asym)
    sym = (ST + ST.T) / 2 - trace * torch.eye(3) / 3

    irreps = tp(S, T)
    assert len(irreps) == 3
    assert torch.allclose(irreps[0], trace)
    assert torch.allclose(irreps[1], asym)
    assert torch.allclose(irreps[2], sym)
