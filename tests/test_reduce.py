import torch

from carten.natural_tensor import NaturalTensors
from carten.reduce import (get_contraction_rule_1, get_dyadic_tensor, get_permutations, reduce_symmetric_tensor, remove_trace, remove_trace_rule, symmetrize,
                           get_contraction_rule_2, )
from carten.test_reduce import get_unique_choose_two
from carten.utils import check_symmetric, check_traceless, letter_index


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


def test_symmetrize(T2, T3, T4):
    for t in [T2, T3, T4]:
        for start_dim in range(2):
            sym = symmetrize(t, start_dim)
            check_symmetric(sym, start_dim)


def test_remove_trace_rule():
    rule, sym = remove_trace_rule(5, 2)
    assert rule == "...aabbc,de,fg->...cdefg"
    assert sym == "abbcc"


def test_remove_trace(T2, T3, T4):
    """
    Test traceless part, not the symmetric part.
    """
    # second rank tensor
    t2 = symmetrize(T2)
    t2 = t2.reshape(1, 1, 3, 3)

    start_dim = 2
    t2 = remove_trace(t2, start_dim)

    assert torch.einsum("...ii", t2) == 0.0

    # third rank tensor
    t3 = symmetrize(T3)
    t3 = t3.reshape(1, 1, 3, 3, 3)

    start_dim = 2
    t3 = remove_trace(t3, start_dim)

    # contract one pair of indices
    num_contract_pair = 1
    delta_indices = get_unique_choose_two(letter_index(3), remove_duplicates=False)
    deltas = delta_indices[num_contract_pair]

    for d in deltas:
        rule = get_contraction_rule_1(d, num_contract_pair)
        v = torch.einsum("..." + rule, t3)
        assert torch.allclose(v, torch.zeros(3), atol=1e-5)

    # fourth rank tensor
    t4 = symmetrize(T4)
    t4 = t4.reshape(1, 1, 3, 3, 3, 3)

    start_dim = 2
    t4 = remove_trace(t4, start_dim)

    # contract one pair of indices
    num_contract_pair = 1
    delta_indices = get_unique_choose_two(letter_index(4), remove_duplicates=False)
    deltas = delta_indices[num_contract_pair]

    for d in deltas:
        rule = get_contraction_rule_1(d, num_contract_pair)
        v = torch.einsum("..." + rule, t4)
        assert torch.allclose(v, torch.zeros(3, 3), atol=1e-4)


def test_get_contraction_rule_1():
    x = get_contraction_rule_1(["ab"], 1)
    assert x == "aa"

    x = get_contraction_rule_1(["ab", "c"], 1)
    assert x == "aac->c"

    x = get_contraction_rule_1(["ac", "b"], 1)
    assert x == "aba->b"

    x = get_contraction_rule_1(["bd", "ac"], 1)
    assert x == "abcb->ac"

    x = get_contraction_rule_1(["ac", "bd"], 2)
    assert x == "abab"

    x = get_contraction_rule_1(["bd", "ace"], 1)
    assert x == "abcbe->ace"

    x = get_contraction_rule_1(["ac", "bd", "e"], 2)
    assert x == "ababe->e"


def test_get_contraction_rule_2():
    x = get_contraction_rule_2(["ab"], 1)
    assert x == "...aa,zy->zy"

    x = get_contraction_rule_2(["ab", "c"], 1)
    assert x == "...aac,zy->zyc"

    x = get_contraction_rule_2(["ac", "b"], 1)
    assert x == "...aba,zy->zby"

    x = get_contraction_rule_2(["bd", "ac"], 1)
    assert x == "...abcb,zy->azcy"

    x = get_contraction_rule_2(["ac", "bd"], 2)
    assert x == "...abab,zy,xw->zxyw"

    x = get_contraction_rule_2(["bd", "ace"], 1)
    assert x == "...abcbe,zy->azcye"

    x = get_contraction_rule_2(["ac", "bd", "e"], 2)
    assert x == "...ababe,zy,xw->zxywe"


def test_get_unique_choose_two():
    def frozen(x):
        """set of frozensets"""
        return set([frozenset(s) for s in x])

    indices = "abc"
    results = get_unique_choose_two(indices)
    assert len(results) == 1

    ref = [["ab", "c"], ["ac", "b"], ["bc", "a"]]
    assert frozen(results[1]) == frozen(ref)

    indices = "abcd"
    results = get_unique_choose_two(indices)
    assert len(results) == 2

    ref = [["ab", "cd"], ["ac", "bd"], ["ad", "bc"]]
    assert frozen(results[1]) == frozen(ref)
    assert frozen(results[2]) == frozen(ref)

    indices = "abcde"

    results = get_unique_choose_two(indices)
    assert len(results) == 2

    ref1 = [
        ["ab", "cde"],
        ["ac", "bde"],
        ["ad", "bce"],
        ["ae", "bcd"],
        ["bc", "ade"],
        ["bd", "ace"],
        ["be", "acd"],
        ["cd", "abe"],
        ["ce", "abd"],
        ["de", "abc"],
    ]
    assert frozen(results[1]) == frozen(ref1)

    ref2 = [
        ["ab", "cd", "e"],
        ["ab", "ce", "d"],
        ["ab", "de", "c"],
        ["ac", "bd", "e"],
        ["ac", "be", "d"],
        ["ac", "de", "b"],
        ["ad", "bc", "e"],
        ["ad", "be", "c"],
        ["ad", "ce", "b"],
        ["ae", "bc", "d"],
        ["ae", "bd", "c"],
        ["ae", "cd", "b"],
        ["bc", "de", "a"],
        ["bd", "ce", "a"],
        ["be", "cd", "a"],
    ]
    assert frozen(results[2]) == frozen(ref2)
