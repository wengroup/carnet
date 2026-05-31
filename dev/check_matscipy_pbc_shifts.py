"""Dev check for matscipy ``neighbour_list`` correctness under mixed PBC.

Background
----------
ASE convention: for any axis ``k`` with ``pbc[k] == False`` there are no
periodic images along ``k``. Every returned neighbour pair on such an axis
must have shift component ``S[:, k] == 0``, and the cell row along ``k`` must
have no effect on the result (it is unused).

What this script does (as of 2026-05-13)
----------------------------------------
Uses a real BCC Fe crystal (2 atoms, ``a = 2.87`` Å, cubic conventional cell
from ``ase.build.bulk``) and ``cutoff = 5.0`` Å. For each of the 8 PBC
combinations in 3D, three checks are run:

  (1) matscipy(original cell) == ASE ``primitive_neighbor_list``(original cell).
  (2) For PBC combos with at least one False axis: shrink the cell rows along
      every non-periodic axis to 0.2× their length and verify that matscipy
      returns the same neighbour list as on the original cell.
      Shrinking (rather than enlarging) is used because, if matscipy
      mistakenly periodicises a False axis, the smaller cell would generate
      many extra image edges and the bug would surface immediately.
  (3) matscipy(perturbed cell) == ASE ``primitive_neighbor_list``(perturbed cell).

Additionally, the fully non-periodic case is also run with ``cell=None`` to
exercise the code path the original bug report used.

Expected behaviour
------------------
All three checks should pass for every PBC combination, and the ``cell=None``
case should produce the same all-zero shifts that ASE produces. Any deviation
is printed as ``[BUG]`` with a one-line ``differs in:`` summary listing which
of ``i``, ``j``, ``S``, ``D`` differs.

Verbose mode
------------
Set the module-level ``VERBOSE`` constant below to ``True`` to also dump the
full ``i``, ``j``, ``S`` arrays for every ``[BUG]`` case.

Run::

    python dev/check_matscipy_pbc_shifts.py
"""

from itertools import product

import numpy as np
from ase.build import bulk
from ase.neighborlist import primitive_neighbor_list
from matscipy.neighbours import neighbour_list

# Knobs.
CUTOFF = 5.0
VERBOSE = False  # set True to dump i/j/S arrays for every [BUG] case


def _bcc_fe():
    """Return (cell, positions) for a 2-atom BCC Fe conventional cell."""
    atoms = bulk("Fe", "bcc", a=2.87, cubic=True)

    return np.array(atoms.cell), atoms.get_positions()


def _sort_edges(i, j, S, D):
    key = np.lexsort((S[:, 2], S[:, 1], S[:, 0], j, i))

    return i[key], j[key], S[key], D[key]


def _run_matscipy(cell, positions, pbc):
    i, j, S, D = neighbour_list(
        "ijSD", pbc=pbc, cell=cell, positions=positions, cutoff=CUTOFF
    )

    return _sort_edges(i, j, S, D)


def _run_ase(cell, positions, pbc):
    ase_cell = np.eye(3) if cell is None else cell
    i, j, S, D = primitive_neighbor_list(
        "ijSD", pbc=pbc, cell=ase_cell, positions=positions, cutoff=CUTOFF
    )

    return _sort_edges(i, j, S, D)


def _perturb_cell(cell, pbc, scale=0.2):
    """Shrink the cell rows along non-periodic axes to ``scale`` of their length.

    The neighbour list along those axes should not depend on these values. We
    shrink (rather than enlarge) so that, if matscipy mistakenly treats the
    axis as periodic, many spurious image edges would appear and the bug shows.
    """
    perturbed = cell.copy().astype(float)
    for k in range(3):
        if not pbc[k]:
            perturbed[k] = scale * cell[k]

    return perturbed


def _equal_lists(a, b):
    i_a, j_a, S_a, D_a = a
    i_b, j_b, S_b, D_b = b

    return (
        i_a.shape == i_b.shape
        and np.array_equal(i_a, i_b)
        and np.array_equal(j_a, j_b)
        and np.array_equal(S_a, S_b)
        and np.allclose(D_a, D_b, atol=1e-10)
    )


def _diff_summary(a, b):
    """Return a short string listing which of (i, j, S, D) differs between a and b."""
    i_a, j_a, S_a, D_a = a
    i_b, j_b, S_b, D_b = b
    diffs = []
    if i_a.shape != i_b.shape:
        diffs.append(f"edge count ({len(i_a)} vs {len(i_b)})")

        return ", ".join(diffs)
    if not np.array_equal(i_a, i_b):
        diffs.append("i")
    if not np.array_equal(j_a, j_b):
        diffs.append("j")
    if not np.array_equal(S_a, S_b):
        n = int(np.any(S_a != S_b, axis=1).sum())
        diffs.append(f"S ({n}/{len(S_a)} rows)")
    if not np.allclose(D_a, D_b, atol=1e-10):
        max_d = float(np.max(np.abs(D_a - D_b)))
        diffs.append(f"D (max |Δ|={max_d:.3g})")

    return ", ".join(diffs) if diffs else "(no diff)"


def _dump(label, lst):
    i, j, S, _ = lst
    print(f"    {label}: i={i.tolist()}  j={j.tolist()}")
    print(f"    {label}: S=\n{S}")


def _check_case(label, cell, positions, pbc):
    print(f"\n--- {label}  pbc={pbc} ---")
    orig = _run_matscipy(cell, positions, pbc)
    ase_orig = _run_ase(cell, positions, pbc)

    # Check 1: matscipy(original cell) agrees with ASE.
    ok1 = _equal_lists(orig, ase_orig)
    if ok1:
        print(f"  [ok] matscipy matches ASE on original cell ({len(orig[0])} edges)")
    else:
        print(
            f"  [BUG] matscipy disagrees with ASE on original cell  "
            f"(matscipy={len(orig[0])} edges, ase={len(ase_orig[0])} edges)"
        )
        print(f"        differs in: {_diff_summary(orig, ase_orig)}")
        if VERBOSE:
            _dump("matscipy", orig)
            _dump("ase     ", ase_orig)

    # Checks 2 and 3: perturb cell rows on non-periodic axes.
    ok2 = ok3 = True
    if cell is not None and any(not p for p in pbc):
        perturbed = _perturb_cell(cell, pbc)
        m_pert = _run_matscipy(perturbed, positions, pbc)
        a_pert = _run_ase(perturbed, positions, pbc)

        # Check 2: matscipy(original) == matscipy(perturbed).
        ok2 = _equal_lists(orig, m_pert)
        if ok2:
            print("  [ok] non-periodic cell rows do not affect matscipy result")
        else:
            print("  [BUG] matscipy neighbour list changes when non-periodic cell")
            print(f"        differs in: {_diff_summary(orig, m_pert)}")
            if VERBOSE:
                _dump("orig    ", orig)
                _dump("perturbd", m_pert)

        # Check 3: matscipy(perturbed) agrees with ASE(perturbed).
        ok3 = _equal_lists(m_pert, a_pert)
        if ok3:
            print(
                f"  [ok] matscipy matches ASE on perturbed cell ({len(m_pert[0])} edges)"
            )
        else:
            print("  [BUG] matscipy disagrees with ASE on perturbed cell")
            print(f"        differs in: {_diff_summary(m_pert, a_pert)}")
            if VERBOSE:
                _dump("matscipy", m_pert)
                _dump("ase     ", a_pert)
    else:
        print("  [skip] no non-periodic axis to perturb")

    return ok1 and ok2 and ok3


def main():
    cell, positions = _bcc_fe()
    print(f"BCC Fe, cell=\n{cell}\npositions=\n{positions}")

    all_ok = True

    print("\n========== cell from ASE bulk('Fe', 'bcc') ==========")
    for pbc in product([False, True], repeat=3):
        all_ok &= _check_case("bcc-Fe", cell, positions, list(pbc))

    print("\n========== cell=None (fully non-periodic) ==========")
    all_ok &= _check_case("cell=None", None, positions, [False, False, False])

    print("\n" + "=" * 50)
    if all_ok:
        print("ALL CHECKS PASSED")
    else:
        print("FAILURES DETECTED — set VERBOSE=True at the top of the file for arrays")


if __name__ == "__main__":
    main()
