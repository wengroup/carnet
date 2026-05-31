"""Correctness tests for ``get_neigh`` (matscipy and ASE backends).

Tests pin down hand-derived expected values on a simple dimer and check
structural invariants on a 2-atom BCC Fe cell.

Both backends (``carnet.data.neighbor`` — matscipy; ``carnet.data.neighbor1``
— ASE) must satisfy every test.
"""

from itertools import product

import numpy as np
import pytest
from ase.build import bulk

from carnet.data import neighbor, neighbor1

BACKENDS = [
    pytest.param(neighbor.get_neigh, id="matscipy"),
    pytest.param(neighbor1.get_neigh, id="ase"),
]


def _dimer():
    """Return (cell, positions, pbc) for a 2-atom dimer.

    Two atoms on the x axis, 1 Å apart, inside a cubic 2×2×2 box. Tests use
    cutoff 1.5 Å. The default ``pbc`` is fully open; tests that need a periodic
    axis override it.
    """
    cell = 2.0 * np.eye(3)
    pos = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])

    return cell, pos, (False, False, False)


def _bcc_fe():
    """Return (cell, positions, pbc) for a 2-atom BCC Fe conventional cell.

    Tests that need a non-default ``pbc`` simply ignore the returned value.
    """
    atoms = bulk("Fe", "bcc", a=2.87, cubic=True)

    return np.array(atoms.cell), atoms.get_positions(), (True, True, True)


def _sort_edges(edge_index, S, D):
    i, j = edge_index
    key = np.lexsort((S[:, 2], S[:, 1], S[:, 0], j, i))

    return np.stack([i[key], j[key]]), S[key], D[key]


def _expected_from_triples(triples):
    """Convert a list of (i, j, shift_xyz, edge_vec_xyz) tuples to sorted arrays."""
    i = np.array([t[0] for t in triples])
    j = np.array([t[1] for t in triples])
    S = np.array([t[2] for t in triples], dtype=np.int64)
    D = np.array([t[3] for t in triples], dtype=float)

    return _sort_edges(np.stack([i, j]), S, D)


# ---------- dimer tests ----------


@pytest.mark.parametrize("get_neigh", BACKENDS)
@pytest.mark.parametrize("use_cell", [True, False], ids=["cell=2I", "cell=None"])
def test_dimer_no_pbc(get_neigh, use_cell):
    cell, pos, pbc = _dimer()
    edge_index, S, D, num_neigh = get_neigh(
        pos, r_cut=1.5, pbc=pbc, cell=cell if use_cell else None
    )
    edge_index, S, D = _sort_edges(edge_index, S, D)

    exp_idx, exp_S, exp_D = _expected_from_triples(
        [
            (0, 1, [0, 0, 0], [1.0, 0.0, 0.0]),
            (1, 0, [0, 0, 0], [-1.0, 0.0, 0.0]),
        ]
    )
    np.testing.assert_array_equal(edge_index, exp_idx)
    np.testing.assert_array_equal(S, exp_S)
    np.testing.assert_allclose(D, exp_D, atol=1e-12)
    np.testing.assert_array_equal(num_neigh, [1, 1])


@pytest.mark.parametrize("get_neigh", BACKENDS)
def test_dimer_pbc_x_only(get_neigh):
    cell, pos, _ = _dimer()
    edge_index, S, D, num_neigh = get_neigh(
        pos, r_cut=1.5, pbc=(True, False, False), cell=cell
    )
    edge_index, S, D = _sort_edges(edge_index, S, D)

    exp_idx, exp_S, exp_D = _expected_from_triples(
        [
            (0, 1, [0, 0, 0], [1.0, 0.0, 0.0]),
            (0, 1, [-1, 0, 0], [-1.0, 0.0, 0.0]),
            (1, 0, [0, 0, 0], [-1.0, 0.0, 0.0]),
            (1, 0, [1, 0, 0], [1.0, 0.0, 0.0]),
        ]
    )
    np.testing.assert_array_equal(edge_index, exp_idx)
    np.testing.assert_array_equal(S, exp_S)
    np.testing.assert_allclose(D, exp_D, atol=1e-12)
    np.testing.assert_array_equal(num_neigh, [2, 2])


@pytest.mark.parametrize("get_neigh", BACKENDS)
def test_dimer_self_interaction(get_neigh):
    cell, pos, pbc = _dimer()
    edge_index, S, D, num_neigh = get_neigh(
        pos, r_cut=1.5, pbc=pbc, cell=cell, self_interaction=True
    )
    edge_index, S, D = _sort_edges(edge_index, S, D)

    exp_idx, exp_S, exp_D = _expected_from_triples(
        [
            (0, 0, [0, 0, 0], [0.0, 0.0, 0.0]),
            (0, 1, [0, 0, 0], [1.0, 0.0, 0.0]),
            (1, 0, [0, 0, 0], [-1.0, 0.0, 0.0]),
            (1, 1, [0, 0, 0], [0.0, 0.0, 0.0]),
        ]
    )
    np.testing.assert_array_equal(edge_index, exp_idx)
    np.testing.assert_array_equal(S, exp_S)
    np.testing.assert_allclose(D, exp_D, atol=1e-12)
    np.testing.assert_array_equal(num_neigh, [2, 2])


# ---------- BCC Fe tests ----------


@pytest.mark.parametrize("get_neigh", BACKENDS)
@pytest.mark.parametrize("pbc", list(product([False, True], repeat=3)))
def test_bcc_fe_invariants(get_neigh, pbc):
    cell, pos, _ = _bcc_fe()
    pbc = tuple(pbc)

    edge_index, S, D, num_neigh = get_neigh(pos, r_cut=5.0, pbc=pbc, cell=cell)

    # (1) shift_vec is zero on every non-periodic axis.
    for k in range(3):
        if not pbc[k]:
            assert np.all(S[:, k] == 0), f"non-zero shift on non-periodic axis {k}"

    # (2) edge_vec is consistent with positions + shift @ cell.
    i, j = edge_index
    expected_D = pos[j] - pos[i] + S @ cell
    np.testing.assert_allclose(D, expected_D, atol=1e-10)

    # (3) num_neigh equals the per-atom count of edges with i == atom.
    counts = np.bincount(i, minlength=len(pos))
    np.testing.assert_array_equal(num_neigh, counts)

    # (4) every distance is within cutoff.
    assert np.all(np.linalg.norm(D, axis=1) <= 5.0 + 1e-10)


@pytest.mark.parametrize("get_neigh", BACKENDS)
@pytest.mark.parametrize("pbc", list(product([False, True], repeat=3)))
def test_bcc_fe_perturb_non_periodic_axes(get_neigh, pbc):
    """Shrinking the cell along non-periodic axes must not change the result."""
    pbc = tuple(pbc)
    if all(pbc):
        pytest.skip("no non-periodic axis to perturb")

    cell, pos, _ = _bcc_fe()
    perturbed = cell.copy()
    for k in range(3):
        if not pbc[k]:
            perturbed[k] = 0.2 * cell[k]

    a = get_neigh(pos, r_cut=5.0, pbc=pbc, cell=cell)
    b = get_neigh(pos, r_cut=5.0, pbc=pbc, cell=perturbed)
    ai, aS, aD = _sort_edges(a[0], a[1], a[2])
    bi, bS, bD = _sort_edges(b[0], b[1], b[2])

    np.testing.assert_array_equal(ai, bi)
    np.testing.assert_array_equal(aS, bS)
    np.testing.assert_allclose(aD, bD, atol=1e-10)
    np.testing.assert_array_equal(a[3], b[3])


@pytest.mark.parametrize("get_neigh", BACKENDS)
def test_bcc_fe_cell_none_matches_explicit_cell(get_neigh):
    cell, pos, _ = _bcc_fe()
    a = get_neigh(pos, r_cut=5.0, pbc=(False, False, False), cell=cell)
    b = get_neigh(pos, r_cut=5.0, pbc=(False, False, False), cell=None)

    ai, aS, aD = _sort_edges(a[0], a[1], a[2])
    bi, bS, bD = _sort_edges(b[0], b[1], b[2])
    np.testing.assert_array_equal(ai, bi)
    np.testing.assert_array_equal(aS, bS)
    np.testing.assert_allclose(aD, bD, atol=1e-10)
    np.testing.assert_array_equal(a[3], b[3])


@pytest.mark.parametrize("get_neigh", BACKENDS)
def test_bcc_fe_self_interaction_adds_n_atoms_edges(get_neigh):
    """Switching self_interaction on must add exactly n_atoms self-edges."""
    cell, pos, pbc = _bcc_fe()

    _, _, _, n0 = get_neigh(pos, r_cut=5.0, pbc=pbc, cell=cell)
    edge_index, S, D, n1 = get_neigh(
        pos, r_cut=5.0, pbc=pbc, cell=cell, self_interaction=True
    )

    assert int(n1.sum()) == int(n0.sum()) + len(pos)
    mask = (edge_index[0] == edge_index[1]) & np.all(S == 0, axis=1)
    assert int(mask.sum()) == len(pos)
    np.testing.assert_allclose(D[mask], 0.0, atol=1e-12)
