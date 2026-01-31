import numpy as np
from ase.neighborlist import primitive_neighbor_list


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
            Ignored if `pbc == False` or pbc == None`.
        self_interaction: Whether to include self-interaction, i.e. an atom being the
            neighbor of itself in the neighbor list. Should be False for most
            applications. Note, an atom will always interact with its periodic image.
            This setting does not control that.

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
        pbc = [pbc] * 3

    if cell is None:
        if not np.any(pbc):
            cell = np.eye(3)  # dummy cell to use
        else:
            raise RuntimeError("`cell` vectors not provided")

    first_idx, second_idx, shift_vec, edge_vec = primitive_neighbor_list(
        "ijSD",
        pbc=pbc,
        cell=cell,
        positions=coords,
        cutoff=r_cut,
        self_interaction=self_interaction,
    )

    # Some atoms do not have neighbors
    n_atoms = len(coords)
    if set(first_idx) != set(range(n_atoms)):
        raise RuntimeError("Some atoms do not have neighbors.")

    # Number of neighbors for each atom
    num_neigh = np.bincount(first_idx, minlength=n_atoms)

    edge_index = np.vstack((first_idx, second_idx))

    return edge_index, shift_vec, edge_vec, num_neigh
