import numpy as np
from matscipy.neighbours import neighbour_list


def get_neigh(
    coords: np.ndarray,
    r_cut: float,
    pbc: bool | tuple[bool, bool, bool] = False,
    cell: np.ndarray | None = None,
    self_interaction: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Create neighbor list for all points in a point cloud.

    Args:
        coords: (N, 3) array of positions, where N is the number of points.
        r_cut: cutoff distance for neighbor finding.
        pbc: Whether to use periodic boundary conditions. If a list of bools, then
            each entry corresponds to a supercell vector. If a single bool, then the
            same value is used for all supercell vectors.
        cell: (3, 3) array of supercell vectors. cell[i] is the i-th supercell vector.
            Rows along non-periodic axes are unused. May be None only when `pbc`
            is fully False.
        self_interaction: If True, include the zero-shift self-edge for every atom
            (i.e. an atom is its own neighbor in the same cell, with shift_vec = 0
            and edge_vec = 0). Should be False for most applications.
            Note: an atom always interacts with its own periodic images
            (entries where i == j and shift_vec != 0); that behaviour is
            governed by `pbc` and `cell`, not by this flag.

    Returns:
        edge_index: (2, num_edges) array of edge indices. The first row contains the
            i atoms (center), and the second row contains the j atoms (neighbor).
        shift_vec: (num_edges, 3) array of shift vectors. The number of cell boundaries
            crossed by the bond between atom i and j. The distance vector between atom
            j and atom i is given by `coords[j] - coords[i] + shift_vec.dot(cell)`.
        edge_vec: (num_edges, 3) array of distance vector.
        num_neigh: (N,) array of the number of neighbors for each atom.
    """
    if isinstance(pbc, bool):
        pbc = np.array([pbc, pbc, pbc])
    else:
        pbc = np.asarray(pbc)

    if cell is None:
        if not pbc.any():
            cell = np.eye(3)  # dummy cell to use
        else:
            raise RuntimeError("`cell` vectors not provided")

    first_idx, second_idx, shift_vec, edge_vec = neighbour_list(
        "ijSD", pbc=pbc, cell=cell, positions=coords, cutoff=r_cut
    )

    # matscipy bug workaround: on non-periodic axes it can emit non-zero shifts S.
    # Zero them out; i, j, D are already correct.
    # TODO: remove this below chunk once the bug is gone.
    #  See dev/check_matscipy_pbc_shifts.py.
    if not pbc.all():
        shift_vec = shift_vec.copy()
        shift_vec[:, ~pbc] = 0

    n_atoms = len(coords)

    # matscipy never emits the (i == i, S = 0) same-cell self-edges. Append
    # them when requested so this backend matches the ASE-based one.
    if self_interaction:
        self_idx = np.arange(n_atoms, dtype=first_idx.dtype)
        first_idx = np.concatenate([first_idx, self_idx])
        second_idx = np.concatenate([second_idx, self_idx])
        shift_vec = np.concatenate(
            [shift_vec, np.zeros((n_atoms, 3), dtype=shift_vec.dtype)]
        )
        edge_vec = np.concatenate(
            [edge_vec, np.zeros((n_atoms, 3), dtype=edge_vec.dtype)]
        )

    # Some atoms do not have neighbors
    if set(first_idx) != set(range(n_atoms)):
        raise RuntimeError("Some atoms do not have neighbors.")

    # Number of neighbors for each atom
    num_neigh = np.bincount(first_idx, minlength=n_atoms)

    edge_index = np.vstack((first_idx, second_idx))

    return edge_index, shift_vec, edge_vec, num_neigh
