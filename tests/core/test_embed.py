from carten.core.embed import embed


def test_embed(NT4):
    for rank in range(5, 10):
        out = embed(NT4, rank)
        assert out.shape == tuple([3] * rank)
