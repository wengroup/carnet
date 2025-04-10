import itertools

import numpy as np

from carten.symbolic.symmetrize import check_symmetry, generate_permutations, symmetrize


def test_generate_permutations():
    # rank-2 symmetric
    symmetry = "ij=ji"
    perms = generate_permutations(symmetry)
    assert set(perms) == set(itertools.permutations([0, 1]))

    # rank-3 partially symmetric
    symmetry = "ijk=ikj"
    perms = generate_permutations(symmetry)
    assert set(perms) == {(0, 1, 2), (0, 2, 1)}

    # rank-3 fully symmetric
    symmetry = "ijk=jik=ikj"
    perms = generate_permutations(symmetry)
    assert set(perms) == set(itertools.permutations([0, 1, 2]))

    # rank-4 partially symmetric (elastic tensor)
    symmetry = "ijkl=jikl=klij"
    perms = generate_permutations(symmetry)
    assert set(perms) == {
        (0, 1, 2, 3),
        (0, 1, 3, 2),
        (1, 0, 2, 3),
        (1, 0, 3, 2),
        (2, 3, 0, 1),
        (2, 3, 1, 0),
        (3, 2, 0, 1),
        (3, 2, 1, 0),
    }

    # rank-4 fully symmetric
    symmetry = "ijkl=jikl=kjil=ljki"
    perms = generate_permutations(symmetry)
    assert set(perms) == set(itertools.permutations([0, 1, 2, 3]))


def test_symmetrize():
    np.random.seed(35)

    t = np.random.randn(3, 3)
    symmetry = "ij=ji"
    out = symmetrize(t, symmetry)
    assert check_symmetry(out, symmetry)

    t = np.random.randn(3, 3, 3)
    symmetry = "ijk=ikj"
    out = symmetrize(t, symmetry)
    assert check_symmetry(out, symmetry)

    symmetry = "ijk=jik=ikj"
    out = symmetrize(t, symmetry)
    assert check_symmetry(out, symmetry)

    t = np.random.randn(3, 3, 3, 3)
    symmetry = "ijkl=jikl=klij"
    out = symmetrize(t, symmetry)
    assert check_symmetry(out, symmetry)

    symmetry = "ijkl=jikl=kjil=ljki"
    out = symmetrize(t, symmetry)
    assert check_symmetry(out, symmetry)
