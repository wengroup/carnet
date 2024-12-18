from carten.core.symmetrize import get_permutations, get_permutations_2


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
