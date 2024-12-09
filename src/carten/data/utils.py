from typing import Union

import torch
from torch import Tensor


@torch.jit.script
def get_edge_vec(
    pos: Tensor,
    shift_vec: Tensor,
    cell: Union[Tensor, None],
    edge_index: Tensor,
    batch: Tensor,
) -> Tensor:
    """
    Create edge vectors for each edge, considering periodic boundary conditions.

    Args:
        pos: (n_nodes, 3) array of atomic positions.
        shift_vec: (n_edges, 3) array of shift vectors. The number of cell boundaries
            crossed by the bond between atom i and j. The distance vector between atom
            j and atom i is given by `pos[j] - pos[i] + shift_vec.dot(cell)`. Ignored
            if cell is None.
        cell: (n_graph*3, 3) array of lattice vectors. If None, then no periodic
            boundary conditions are considered.
        edge_index: (2, n_edges) array of edge indices, where the first row is the
            source node index and the second row is the target node index.
        batch: (n_nodes,) array of batch indices, giving the graph that each node
            belongs to.

    Returns:
        edge_vec: (n_edges, 3) array of distance vectors.
    """
    i_idx = edge_index[0]
    j_idx = edge_index[1]

    edge_vec = pos[j_idx] - pos[i_idx]

    if cell is not None:
        # create a cell for each edge, shape (num_edges, 3, 3)
        expanded_cell = cell.reshape(-1, 3, 3)[batch[i_idx]]

        edge_vec = edge_vec + torch.einsum("ni, nij -> nj", shift_vec, expanded_cell)

    return edge_vec
