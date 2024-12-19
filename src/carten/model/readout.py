"""Readout layer."""
import torch
from torch import Tensor, nn

from carten.module.mlp import MLP
from carten.module.scatter import scatter


class StructureScalar(nn.Module):
    """Get a scalar output for each atomic configuration.

    For a model with multiple layers, the scalar outputs of all layers are passed to
    this module. This module then computes a final scalar output for each atomic
    configuration, achieved by the steps:
    1. The scalar atom feats of each layer (but the last) are multiplied by a weight
       matrix.
    2. The scalar atom feats of the last layer are passed through an MLP.
    3. The scalar atom feats of all layers are summed to get the atom feats.
    4. The scalar atom feats of each atomic configuration are summed to get the scalar
       output for each atomic configuration.

    Args:
        num_layers: Number of layers in the main model.
        in_features: Size of the input features.
        hidden_features: Size of the features for the hidden layers of MLP to process
            the scalar atom feats of the last layer. If a list, it provides the hidden
            layer sizes of the MLP. If an integer, it is interpreted as the number of
            hidden layers, and the hidden layer sizes are set to in_features.
        atomic_shift: Shift of the output. See `atomic_scale`.
        atomic_scale: Scale of the output. The atomic shift and scale are used to
            transform the output. The final output for each atomic configuration is
            computed as y = x*scale + shift, where x result of the above four steps.
            If `atomic_shift` and `atomic_scale` is None, no transform is applied.
            If a scalar tensor is provided for `atomic_shift` and for `atomic_scale`,
            then it is then it is used for all atom types.
            If a tensor of shape (n_atom_types,) is provided for `atomic_shift` and
                `atomic_scale`, then it is applied to each atom type separately.
    """

    def __init__(
        self,
        num_layers: int,
        in_features: int,
        hidden_features: list[int] | int,
        atomic_shift: Tensor = None,
        atomic_scale: Tensor = None,
    ):
        super().__init__()

        self.num_layers = num_layers

        self.register_buffer(
            "atomic_shift", atomic_shift if atomic_shift is not None else None
        )
        self.register_buffer(
            "atomic_scale", atomic_scale if atomic_scale is not None else None
        )

        # Linear mapping for atom features in early layers
        self.out_layers = nn.ModuleList(
            [nn.Linear(in_features, 1) for _ in range(num_layers - 1)]
        )

        # MLP for atom features in the last layer
        if isinstance(hidden_features, int):
            hidden_features = [in_features for _ in range(hidden_features)]
        self.out_layers.append(
            MLP(in_features, 1, hidden_features, out_activation=False)
        )

    def forward(
        self, atom_feats: list[Tensor], atom_type: Tensor, num_atoms: Tensor
    ) -> Tensor:
        """
        Args:
            atom_feats: list of scalar atomic features, each of shape (n_atoms, F, 1),
                where `F` is the channel dimension.
            atom_type: The atomic type of each atom. Shape (n_atoms,).
            num_atoms: The number of atoms in each atomic configuration.
                Shape (n_atoms,)

        Returns:
            Total energy of each configuration. Shape (n_config,).
        """
        assert len(atom_feats) == self.num_layers, "Number of layers mismatch."

        V = torch.zeros(1, dtype=atom_feats[0].dtype, device=atom_feats[0].device)
        for i, fn in enumerate(self.out_layers):
            V = V + fn(atom_feats[i].squeeze(-1)).view(-1)  # shape (n_atoms,)

        # Normalization
        if self.atomic_scale is not None:
            if self.atomic_scale.ndim == 0:
                V = V * self.atomic_scale
            else:
                V = V * self.atomic_scale[atom_type]

        if self.atomic_shift is not None:
            if self.atomic_shift.ndim == 0:
                V = V + self.atomic_shift
            else:
                V = V + self.atomic_shift[atom_type]

        # Output of each configuration; (num_config,)
        out = scatter(V, torch.repeat_interleave(num_atoms), reduce="sum", dim=0)

        return out


class AtomicVector(nn.Module):
    """Output module to predict a vector quantity for each atom.

    The shape of the input is (n_u, n_atoms, 3), where n_u denotes the batch dimension
    of the radial degree u (i.e. feature dimension), for each atom, the feature is
    (n_u, 3), i.e. a set of 3-vectors, v1, v2, ... v_{n_u}.

    The output vector v for an atom is obtained via a linear combination of all the
    vectors, v = LinearCombination(v1, v2, ... v_{n_u})
    """

    def __init__(self, num_layers: int, in_features: int):
        super().__init__()

        self.num_layers = num_layers

        # last layer
        self.layer = nn.Linear(in_features, 1, bias=False)

    def forward(
        self, atom_feats: Tensor, atom_type: Tensor, num_atoms: Tensor
    ) -> Tensor:
        """
        Args:
            atom_feats: atomic features of shape (n_u, n_atoms, 3), where n_u
                denotes the batch dimension of the radial degree u (i.e. feature
                dimension), and the 3 is the vector dimension.

        Returns:
            Atomic vectors, tensor of shape (n_atoms, 3).
        """

        V = self.layer(atom_feats.permute(1, 2, 0)).squeeze(-1)  # shape (n_atoms, 3)

        return V
