import pytest

from carnet.data.data import get_edge_vec_batch

# from carnet.model.carnet import CarNet
#
#
# @pytest.fixture
# def model():
#     m = CarNet(num_atom_types=2, max_v=2, max_u=2, num_average_neigh=5.0)
#     return m
#
#
# def test_datapipe(model, dataloader):
#     for batch in dataloader:
#         edge_vector = get_edge_vec(
#             batch.pos, batch.shift_vec, batch.cell, batch.edge_index, batch.batch
#         )
#
#         energy = model(edge_vector, batch.edge_index, batch.atom_type, batch.num_atoms)
#         assert energy.shape == (batch.num_graphs,)
#
#         forces = model.compute_forces(energy, batch.pos)
#         assert forces.shape == (batch.num_nodes, 3)
