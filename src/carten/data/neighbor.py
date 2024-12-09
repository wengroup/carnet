import numpy as np
from ase.neighborlist import primitive_neighbor_list


def get_neigh(
    coords: np.ndarray,
    r_cut: float,
    pbc: bool | tuple[bool, bool, bool] = False,
    cell: np.ndarray | None = None,
    self_interaction: bool = False,
    periodic_self_interaction: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create neighbor list for all points in a point cloud.

    Args:
        coords: (N, 3) array of positions, where N is the number of points.
        r_cut: cutoff distance for neighbor finding.
        pbc: Whether to use periodic boundary conditions. If a list of bools, then
            each entry corresponds to a supercell vector. If a single bool, then the
            same value is used for all supercell vectors.
        cell: (3, 3) array of supercell vectors. cell[i] is the i-th supercell vector.
            Ignored if `pbc == False` or pbc == None`.
        self_interaction: Whether to include self-interaction, i.e. an atom being the
            neighbor of itself in the neighbor list. Should be False for most
            applications.
        periodic_self_interaction: Whether to include interactions of an atom with its
            periodic images. Should be True for most applications.

    Returns:
        edge_index: (2, num_edges) array of edge indices. The first row contains the
            i atoms (center), and the second row contains the j atoms (neighbor).
        shift_vec: (num_edges, 3) array of shift vectors. The number of cell boundaries
            crossed by the bond between atom i and j. The distance vector between atom
            j and atom i is given by `coords[j] - coords[i] + shift_vec.dot(cell)`.
        num_neigh: (N,) array of the number of neighbors for each atom.
    """
    if isinstance(pbc, bool):
        pbc = [pbc] * 3

    if not np.any(pbc):
        self_interaction = False
        periodic_self_interaction = False

    if cell is None:
        if not np.any(pbc):
            cell = np.eye(3)  # dummy cell to use
        else:
            raise RuntimeError("`cell` vectors not provided")

    first_idx, second_idx, shift_vec = primitive_neighbor_list(
        "ijS",
        pbc=pbc,
        cell=cell,
        positions=coords,
        cutoff=r_cut,
        self_interaction=periodic_self_interaction,
    )

    # remove self interactions
    if periodic_self_interaction and (not self_interaction):
        bad_edge = first_idx == second_idx
        bad_edge &= np.all(shift_vec == 0, axis=1)
        keep_edge = ~bad_edge
        if not np.any(keep_edge):
            raise RuntimeError(
                "After removing self interactions, no edges remain in this system."
            )
        first_idx = first_idx[keep_edge]
        second_idx = second_idx[keep_edge]
        shift_vec = shift_vec[keep_edge]

    # number of neighbors for each atom
    num_neigh = np.bincount(first_idx)

    # Some atoms with large index may not have neighbors due to the use of bincount.
    # As a concrete example, suppose we have 5 atoms and first_idx is [0,1,1,3,3,3,3],
    # then bincount will be [1, 2, 0, 4], which means atoms 0,1,2,3 have 1,2,0,4
    # neighbors respectively. Although atom 2 is handled by bincount, atom 4 is not.
    # The below part is to make this work.
    if len(num_neigh) != len(coords):
        extra = np.zeros(len(coords) - len(num_neigh), dtype=int)
        num_neigh = np.concatenate((num_neigh, extra))

    edge_index = np.vstack((first_idx, second_idx))

    return edge_index, shift_vec, num_neigh
