"""CARTEN layer module."""
from line_profiler import profile
from torch import Tensor, nn

from .atomic_moment import AtomicMoment, AtomicMoment2
from .hyper_moment import HyperMoment
from .linear import SlicedLinearMap


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
        atomic_moment_mode: str = "vanilla",
    ):
        """

        Args:
            F:
            L1:
            L2:
            L3:
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
        self.atomic_moment_mode = atomic_moment_mode

        # TODO, we might not need this, given that we perform the mixing at the end
        #  This might be beneficial for the first layer.
        # Kernel for mixing atom feats across channel, separate for each rank
        self.linear_channel_input = SlicedLinearMap(
            F, F, [3**l for l in range(L1 + 1)], bias=True
        )

        if atomic_moment_mode == "vanilla":
            AM = AtomicMoment
        elif atomic_moment_mode in ["variant1", "variant2"]:
            AM = AtomicMoment2
        else:
            raise ValueError(f"Unknown atomic_moment_mode: {atomic_moment_mode}")
        self.atomic_moment = AM(
            F=F,
            L1=L1,
            L2=L2,
            L3=L3,
            num_atom_types=num_atom_types,
            num_average_neigh=num_average_neigh,
            max_chebyshev_degree=max_chebyshev_degree,
            radial_mlp_hidden_layers=radial_mlp_hidden_layers,
            r_cut=r_cut,
            mode=atomic_moment_mode,
        )

        # Kernel for mixing channel of atomic moment, separate for each rank
        self.linear_channel_atomic = SlicedLinearMap(
            F, F, [3**l for l in range(L3 + 1)], bias=True
        )

        self.hyper_moment = HyperMoment(
            F=self.F, L=self.L3, max_out_L=self.max_out_L, max_degree=self.max_degree
        )

        # Kernel for mixing channel of hyper moment, separate for each rank
        self.linear_channel_hyper = SlicedLinearMap(
            F, F, [3**l for l in range(self.max_out_L + 1)], bias=True
        )

        # Kernel for mixing channel of input atom feats, separate for each rank
        if mix_atom_feats_across_channel:
            # Only do it for the ranks that exist in both the input atom feats and the
            # output hyper moment
            self.min_max_out_L_L1 = min(self.max_out_L, self.L1)
            self.linear_channel_feats = SlicedLinearMap(
                F, F, [3**l for l in range(self.min_max_out_L_L1 + 1)], bias=True
            )
        else:
            self.register_buffer("linear_channel_feats", None)

    @profile
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

        # TODO, This seems not needed, we have too many mixing
        # Mixing input atom feats across channel
        feats = self.linear_channel_input(atom_feats)  # (Na, F, T1)

        # Get atomic moments; (Na, F, T3)
        am = self.atomic_moment(edge_vector, edge_idx, atom_type, feats)

        # Mix atomic moments across channel
        am_mixed = self.linear_channel_atomic(am)  # (Na, F, T3)

        # Get hyper moments; (Na, F, T')
        hm = self.hyper_moment(am_mixed)

        # Mix hyper moments across channel
        hm_mixed = self.linear_channel_hyper(hm)  # (Na, F, T')

        # Mix input atom feats across channel and add to the output
        if self.linear_channel_feats is not None:
            size = int((3 ** (self.min_max_out_L_L1 + 1) - 1) // 2)
            feats_skip_connection = self.linear_channel_feats(atom_feats[..., :size])
            hm_mixed[..., :size] += feats_skip_connection

        return hm_mixed
