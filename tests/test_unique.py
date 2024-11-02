from carten.unique import epsilon_T


def test_epsilon_T():
    out = epsilon_T(1, "ijkl")
    assert out == ["23kl", "-32kl"]

    out = epsilon_T(2, "kimjn")
    assert out == ["k3m1n", "-k1m3n"]
