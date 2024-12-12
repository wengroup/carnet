"""CARTEN layer module."""
import torch
from torch import Tensor, nn

from .atomic_moment import AtomicMoment
from .hyper_moment import HyperMoment
from .linear import LinearMap


class Layer(nn.Module):
    """
    A layer consisting of an atomic moment module followed by a hyper moment module.
    """

    def __init__(
        self,
        F: int,
        L1: int,
        L2: int,
        L3: int,
        num_atom_types: int,
        num_average_neigh: float,
        max_chebyshev_degree: int = 8,
        r_cut: float = 5.0,
        radial_mlp_hidden_layers: int | list[int] = 2,
        max_out_L: int = None,
        max_degree: int = None,
        mix_atom_feats_across_channel: bool = True,
    ):
        """

        Args:
            num_atom_types:
            num_average_neigh:
            max_chebyshev_degree:
            r_cut:
            radial_mlp_hidden_layers: if list of int, this gives the size of each hidden
                layer in the MLP that is applied to the radial basis functions. If int,
                this gives the number of hidden layers, and the size of each hidden
                layer is set to `F`, the channel dimension.
            max_out_L: Max rank for the output feature tensor. If None, set to L.
            max_degree: Max correlation degree of the hyper moment feature tensor.
                If None, set to L.
            mix_atom_feats_across_channel: whether to mix the channel of the input atom
                feats and then add to the output hyper moment.

        """

        super().__init__()

        self.F = F
        self.L1 = L1
        self.L2 = L2
        self.L3 = L3
        self.num_atom_types = num_atom_types
        self.num_average_neigh = num_average_neigh
        self.max_chebyshev_degree = max_chebyshev_degree
        self.r_cut = r_cut
        self.radial_mlp_hidden_layers = radial_mlp_hidden_layers
        self.max_out_L = L3 if max_out_L is None else max_out_L
        self.max_degree = L3 if max_degree is None else max_degree
        self.mix_atom_feats_across_channel = mix_atom_feats_across_channel

        # TODO, we might not need this, given that we perform the mixing at the end
        #  This might be beneficial for the first layer.
        # Kernel for mixing atom feats across channel, separate for each rank
        self.linear_channel_input = nn.ModuleList(
            [LinearMap(F, F) for _ in range(L1 + 1)]
        )

        self.atom_moment = AtomicMoment(
            F=F,
            L1=L1,
            L2=L2,
            L3=L3,
            num_atom_types=num_atom_types,
            num_average_neigh=num_average_neigh,
            max_chebyshev_degree=max_chebyshev_degree,
            radial_mlp_hidden_layers=radial_mlp_hidden_layers,
            r_cut=r_cut,
        )

        self.hyper_moment = HyperMoment(
            F=self.F, L=self.L3, max_out_L=self.max_out_L, max_degree=self.max_degree
        )

        # Kernel for mixing channel of hyper moment, separate for each rank
        self.linear_channel_hyper = nn.ModuleList(
            [LinearMap(F, F) for _ in range(self.max_out_L + 1)]
        )

        # Kernel for mixing channel of input atom feats, separate for each rank
        if mix_atom_feats_across_channel:
            self.linear_channel_feats = nn.ModuleList(
                [LinearMap(F, F) for _ in range(self.max_out_L + 1)]
            )
        else:
            self.register_buffer("linear_channel_feats", None)

    def forward(
        self,
        edge_vector: Tensor,
        edge_idx: Tensor,
        atom_type: Tensor,
        atom_feats: Tensor,
    ) -> Tensor:
        """

        Args:
            edge_vector: Edge vectors. Shape (n_edges, 3).
            edge_idx: Indices of center and neighbor atoms, that form the edges.
                Shape (2, n_edges). The first row is the center atom indices, and the
                second row is the neighbor atom indices.
            atom_type: Atom types. Shape (Na,), where Na is the number of atoms.
            atom_feats: shape (Na, F, T1), where Na is the number of atoms, F is the
                number of features, and T1 = (3**(L1+1)-1)//2 is the tensor dim.

        Returns:
            Updated atom feats. Shape (Na, F, T'), where T' is the number of tensor
            components, determined by max_out_L.
        """
        # # mix atom feats across radial channel
        # feats: dict[int, Tensor] = {}
        # for v, f in atom_feats.items():
        #     # Make indexing ModuleDict work
        #     # See https://github.com/pytorch/pytorch/issues/68568
        #     fn: JITInterface = self.linear_channel_input[str(v)]
        #     feats[v] = fn.forward(f)
        #
        # am = self.atom_moment(edge_vector, edge_idx, atom_type, feats)
        #
        # hm = self.hyper_moment(am)  # {v: (n_u, n_atoms, 3, 3, ...)}}
        #
        # # mix radial channel of hyper moment
        # # {v: {n_u, n_atoms, 3, 3, ...)}}
        # # hm = {rank: self.linear_channel_hyper[str(rank)](m) for rank, m in hm.items()}
        #
        # for rank, m in hm.items():
        #     fn: JITInterface = self.linear_channel_hyper[str(rank)]
        #     hm[rank] = fn.forward(m)
        #
        # out = hm
        #
        # # mix radial channel of input atom feats and add to the output
        # if self.mix_atom_feats_across_channel:
        #     max_rank = min(self.max_atom_feats_rank, self.max_out_L)
        #     for rank in range(max_rank + 1):
        #         fn: JITInterface = self.linear_channel_feats[str(rank)]
        #         out[rank] = out[rank] + fn.forward(feats[rank])
        #
        # return out

        # Mixing input atom feats across channel
        feats = []
        start = 0
        for l, fn in zip(range(self.L1 + 1), self.linear_channel_input):
            end = start + 3**l
            feats.append(fn(atom_feats[..., start:end]))
            start = end
        feats = torch.cat(feats, dim=-1)  # (Na, F, T1)

        # Get atomic moments; (Na, F, T3)
        am = self.atom_moment(edge_vector, edge_idx, atom_type, feats)

        # TODO, move mixing of atomic moment across channel from atomic_moment.py to here

        # Get hyper moments; (Na, F, T')
        hm = self.hyper_moment(am)

        # TODO, we have so many such mixing, create a function for it and share
        # Mix hyper moments across channel
        start = 0
        for l, fn in zip(range(self.max_out_L + 1), self.linear_channel_hyper):
            end = start + 3**l
            # TODO, inplace change is not OK for backprop
            hm[..., start:end] = fn(hm[..., start:end])
            start = end

        # TODO, if we want to use this, we need to ensure max_out_L < L1, because
        #  atom_feats can be smaller, e.g. for the first layer, it only consists of
        #  scalar features.
        #
        # TODO, we need to modify the definition of self.linear_channel_feats as:
        #  self.linear_channel_feats =
        #  nn.ModuleList([LinearMap(F, F) for _ in range(min(self.max_out_L, self.L1) + 1)])
        #  and modify the below range accordingly.

        # Mix input atom feats across channel and add to the output
        if self.linear_channel_feats is not None:
            start = 0
            for l, fn in zip(range(self.max_out_L + 1), self.linear_channel_feats):
                end = start + 3**l
                # TODO, inplace change is not OK for backprop
                hm[..., start:end] = hm[..., start:end] + fn(atom_feats[..., start:end])
                start = end

        return hm
