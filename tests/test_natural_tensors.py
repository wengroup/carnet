import itertools

import torch

from carten.natural_tensor import (
    NaturalTensors,
    flatten_tensor_dims,
    get_contraction_rule_1,
    get_contraction_rule_2,
    get_unique_choose_two,
    remove_trace,
    symmetrize,
    unflatten_tensor_dims,
)
from carten.signature import Signature
from carten.utils import get_trace, letter_index


def test_NaturalTensors(
    NT0,
    NT1,
    NT2,
    NT3,
    T0_chunk,
    T1_chunk,
    T2_chunk,
    T3_chunk,
    T0_shaped_chunk,
    T1_shaped_chunk,
    T2_shaped_chunk,
    T3_shaped_chunk,
):
    nt = NaturalTensors(
        signature=Signature([(2, 2), (2, 1), (2, 1), (2, 3), (2, 0)]),
        data=torch.cat([T2_chunk, T1_chunk, T1_chunk, T3_chunk, T0_chunk], dim=-1),
    )

    assert nt.signature == [(2, 2), (2, 1), (2, 1), (2, 3), (2, 0)]
    assert nt.num_chunks == 5
    assert nt.min_rank == 0
    assert nt.max_rank == 3

    # simplify
    nt = nt.simplify()
    assert nt.signature == [(2, 2), (4, 1), (2, 3), (2, 0)]
    assert nt.num_chunks == 4

    # regroup()
    regrouped = nt.regroup()
    assert torch.allclose(
        regrouped.data,
        torch.cat([T0_chunk, T1_chunk, T1_chunk, T2_chunk, T3_chunk], dim=-1),
    )

    # .chunks
    for t1, t2 in zip(
        nt.chunks,
        [T2_chunk, torch.cat((T1_chunk, T1_chunk), dim=-1), T3_chunk, T0_chunk],
    ):
        assert torch.allclose(t1, t2)

    # .shaped_chunks
    for t1, t2 in zip(
        nt.shaped_chunks,
        [
            T2_shaped_chunk,
            torch.cat((T1_shaped_chunk, T1_shaped_chunk), dim=-2),
            T3_shaped_chunk,
            T0_shaped_chunk,
        ],
    ):
        assert torch.allclose(t1, t2)

    # from_sequence()
    nt_seq = NaturalTensors.from_sequence([NT2, NT1, NT1, NT3, NT0], check=True)
    assert nt_seq.signature == [(1, 2), (2, 1), (1, 3), (1, 0)]
    assert nt.num_chunks == 4
    assert nt.min_rank == 0
    assert nt.max_rank == 3

    # from_chunks()
    sig = Signature([(2, 1), (2, 0), (2, 2)])
    nt_ch = NaturalTensors.from_chunks(
        signature=sig, data=[T1_chunk, T0_chunk, T2_chunk], check=True
    )

    assert nt_ch.signature == sig
    assert torch.allclose(nt_ch.data, torch.cat((T1_chunk, T0_chunk, T2_chunk), dim=-1))
    regrouped = nt_ch.regroup()
    assert torch.allclose(
        regrouped.data, torch.cat((T0_chunk, T1_chunk, T2_chunk), dim=-1)
    )

    # from shaped chunks
    sig = Signature([(2, 1), (2, 0), (2, 2)])
    nt_sc = NaturalTensors.from_shaped_chunks(
        signature=sig,
        data=[T1_shaped_chunk, T0_shaped_chunk, T2_shaped_chunk],
        check=True,
    )
    assert nt_sc.signature == sig
    assert torch.allclose(
        nt_sc.data,
        torch.cat(
            (
                flatten_tensor_dims(T1_shaped_chunk, 2, 1),
                flatten_tensor_dims(T0_shaped_chunk, 2, 0),
                flatten_tensor_dims(T2_shaped_chunk, 2, 2),
            ),
            dim=-1,
        ),
    )
    regrouped = nt_sc.regroup()
    assert torch.allclose(
        regrouped.data,
        torch.cat(
            (
                flatten_tensor_dims(T0_shaped_chunk, 2, 0),
                flatten_tensor_dims(T1_shaped_chunk, 2, 1),
                flatten_tensor_dims(T2_shaped_chunk, 2, 2),
            ),
            dim=-1,
        ),
    )


def test_NaturalTensors_property(T0_chunk, T2_chunk, T0_shaped_chunk, T2_shaped_chunk):
    sig = Signature([(2, 0), (2, 2)])
    nt = NaturalTensors.from_chunks(signature=sig, data=[T0_chunk, T2_chunk])

    assert nt.dtype == T0_chunk.dtype
    assert nt.device == T0_chunk.device

    assert nt.leading_dim == 1
    assert nt.leading_shape == (1,)

    assert nt.get_chunk(0).shape == T0_chunk.shape
    assert nt.get_chunk(1).shape == T2_chunk.shape

    assert nt.get_shaped_chunk(0).shape == T0_shaped_chunk.shape
    assert nt.get_shaped_chunk(1).shape == T2_shaped_chunk.shape


def test_get_trace(T2, T3):
    trace = get_trace(T2, i=0, j=1)
    assert torch.allclose(trace, torch.tensor([12.0]))

    trace = get_trace(T3, i=0, j=1)
    assert torch.allclose(trace, torch.tensor([36.0, 39.0, 42.0]))

    trace = get_trace(T3, i=1, j=2)
    assert torch.allclose(trace, torch.tensor([12.0, 39.0, 66.0]))


def test_symmetrize(T2, T3, T4):
    for t in [T2, T3, T4]:
        for start_dim in range(2):
            sym_t = symmetrize(t, start_dim)
            for p in itertools.permutations(range(start_dim, t.ndim)):
                p = list(range(start_dim)) + list(p)
                assert torch.allclose(sym_t, sym_t.permute(*p))


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


def test_flatten_tensor_dims(T0_chunk, T0_shaped_chunk, T2_chunk, T2_shaped_chunk):
    flattened0 = flatten_tensor_dims(T0_shaped_chunk, 2, 0, check=True)
    assert flattened0.shape == T0_chunk.shape
    assert torch.allclose(flattened0, T0_chunk)

    flattened2 = flatten_tensor_dims(T2_shaped_chunk, 2, 2, check=True)
    assert flattened2.shape == T2_chunk.shape
    assert torch.allclose(flattened2, T2_chunk)


def test_unflatten_tensor_dims(T0_chunk, T0_shaped_chunk, T2_chunk, T2_shaped_chunk):
    shaped0 = unflatten_tensor_dims(T0_chunk, 2, 0)
    assert shaped0.shape == T0_shaped_chunk.shape
    assert torch.allclose(shaped0, T0_shaped_chunk)

    shaped2 = unflatten_tensor_dims(T2_chunk, 2, 2)
    assert shaped2.shape == T2_shaped_chunk.shape
    assert torch.allclose(shaped2, T2_shaped_chunk)
