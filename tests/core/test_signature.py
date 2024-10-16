from carten.core.signature import Signature


def test_Signature():
    s = Signature([(5, 0), 2, (6, 4)])

    assert (
        s
        == Signature([(5, 0), (1, 2), (6, 4)])
        == Signature.from_str("  5x   0 +    2   + 6   x  4   ")
        == Signature.from_str("  5x   0 +    1x2 + 6   x  4   ")
        == Signature.from_ranks([0, 0, 0, 0, 0, 2, 4, 4, 4, 4, 4, 4])
    )
    assert s.dim == 5 + 1 * 9 + 6 * 81

    s = Signature([(10, 3), (11, 1), (12, 2), (13, 2), (14, 1)])
    assert s.simplify() == [(10, 3), (11, 1), (25, 2), (14, 1)]

    ordered = s.reorder()
    assert ordered.signature == [(11, 1), (14, 1), (12, 2), (13, 2), (10, 3)]
    assert s.regroup() == s.reorder().signature.simplify()
    assert s.regroup() == [(25, 1), (25, 2), (10, 3)]

    s = Signature([(2, 2), (2, 0), (2, 1)])
    chunk_dims = [18, 2, 6]
    assert chunk_dims == s.chunk_dims

    ordered = s.reorder()
    assert ordered.signature == [(2, 0), (2, 1), (2, 2)]
    assert ordered.sig_perm == [1, 2, 0]
    assert ordered.data_perm == (
        [18 + i for i in range(2)]
        + [18 + 2 + i for i in range(6)]
        + [i for i in range(18)]
    )
