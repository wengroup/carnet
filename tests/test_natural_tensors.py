import torch

from carnet.legacy.natural_tensor import (
    NaturalTensors,
    flatten_tensor_dims,
    unflatten_tensor_dims,
)
from carnet.legacy.signature import Signature


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
