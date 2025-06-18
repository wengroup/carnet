"""Readout layer."""

import torch
from torch import Tensor, nn

from carten.module.linear import LinearMap
from carten.module.mlp import MLP
from carten.module.scatter import scatter
from carten.utils import BufferDict


class StructureScalar(nn.Module):
    """Get a scalar output for each configuration.

    For a model with multiple layers, the scalar atomic feats of all layers are passed
    to this module. This module then computes a final scalar output for each atomic
    configuration, achieved by the steps:
    1. The scalar atom feats of each layer (but the last) are multiplied by a weight
       matrix to combine the features across the channel dimension.
    2. The scalar atom feats of the last layer are passed through an MLP.
    3. The feats of all layers are summed to get the total atom feats.
    4. The total atom feats are summed to get the scalar output for each configuration.

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
        self.in_features = in_features
        self.hidden_features = hidden_features

        self.register_buffer("atomic_shift", atomic_shift)
        self.register_buffer("atomic_scale", atomic_scale)

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
        # TODO, add element_bias, as in AtomicTensor, if needed.

    def forward(
        self, atom_feats: list[Tensor], atom_type: Tensor, num_atoms: Tensor
    ) -> Tensor:
        """
        Args:
            atom_feats: list of scalar atomic features, each of shape (n_atoms, F, 1),
                where `F` is the channel dimension.
            atom_type: The atomic type of each atom. Shape (n_atoms,).
            num_atoms: The number of atoms in each atomic configuration.
                Shape (n_config,)

        Returns:
            Total energy of each configuration. Shape (n_config,).
        """
        assert len(atom_feats) == self.num_layers, "Incorrect number of atom feats."

        V = torch.zeros(
            torch.sum(num_atoms), dtype=atom_feats[0].dtype, device=atom_feats[0].device
        )
        for i, fn in enumerate(self.out_layers):
            V += fn(atom_feats[i].squeeze(-1)).view(-1)  # shape (n_atoms,)

        # Normalization
        if self.atomic_scale is not None:
            if self.atomic_scale.ndim == 0:
                V *= self.atomic_scale
            else:
                V *= self.atomic_scale[atom_type]

        if self.atomic_shift is not None:
            if self.atomic_shift.ndim == 0:
                V += self.atomic_shift
            else:
                V += self.atomic_shift[atom_type]

        # Output of each configuration; (num_config,)
        out = scatter(V, torch.repeat_interleave(num_atoms), reduce="sum", dim=0)

        return out


class AtomicTensor(nn.Module):
    """Get a tensor output for each atom in a configuration.

    For a model with multiple layers, the atomic feats of all layers are passed to this
    module. This module then:
    1. Selects the corresponding natural tensors according to `atomic_selector`.
    2. Linearly maps them (via the channel dim) to get the atomic natural tensors.


    Args:
        num_layers: Number of layers in the main model.
        in_features: Size of the input features, namely the channel dimension F.
        hidden_features: Size of the features for the hidden layers of MLP to process
            the scalar atom feats of the last layer. If a list, it provides the hidden
            layer sizes of the MLP. If an integer, it is interpreted as the number of
            hidden layers, and the hidden layer sizes are set to in_features.
            If the output has no rank-0 natural tensor, this argument is ignored.
        output_signature: A dictionary {l: n_l} that specifies the natural tensor
            components to output for each atomic configuration. The key `l` gives
            the rank of the natural tensor, and the value `n_l` gives the number of
            rank-l natural tensor to output. For example, a dielectric tensor is a
            symmetric rank-2 tensor, which can be decomposed as 1 rank-0  and 1 rank-2
            natural tensors. To model the dielectric tensor, the output_signature
            should be {0: 1, 1: 1, 2: 1}. As another example, the elastic tensor is a
            rank-4 tensor, which can be decomposed as 2 rank-0, 2 rank-2, and 1 rank-4
            natural tensors. To model the elastic tensor, the output_signature should
            be {0: 2, 2: 2, 4: 1}.
        num_atom_types: Number of atomic types.
        target_shift: A dictionary {l: shift} that specifies the shift to apply to
            the output of the model before computing the loss. Used together with
            target_scale.
        target_scale: A dictionary {l: scale} that specifies the scale to apply to
            the output of the model before computing the loss. Used together with
            target_shift. y = scale*z + shift, where z is the output of the
            network, and y is the predicted target.
        element_bias: If True, a separate bias is learned for each atomic type and
            added to the scalar part of the model.
        num_atom_feats: Number of atomic features to expect in the forward pass. If
            None, it is set to `num_layers`, indicating that the atomic features of
            all layers are passed to this module.
    """

    def __init__(
        self,
        num_layers: int,
        in_features: int,
        hidden_features: list[int] | int,
        output_signature: dict[int, int],
        num_atom_types: int,
        target_shift: dict[int, Tensor] = None,
        target_scale: dict[int, Tensor] = None,
        element_bias: bool = True,
        num_atom_feats: int = None,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.output_signature = output_signature
        self.num_atom_types = num_atom_types

        self.num_atom_feats = (
            num_atom_feats if num_atom_feats is not None else num_layers
        )

        self.kernel = nn.ModuleDict()
        self.slice = dict()

        for l, n in output_signature.items():
            # For scalar, if any, using an MLP
            if l == 0:
                if isinstance(hidden_features, int):
                    hidden_features = [in_features for _ in range(hidden_features)]
                self.kernel[str(l)] = MLP(
                    in_features * self.num_atom_feats,
                    n,
                    hidden_features,
                    out_activation=False,
                )
            # For others, using a linear map
            else:
                self.kernel[str(l)] = LinearMap(in_features * self.num_atom_feats, n)

            start = (3**l - 1) // 2
            end = (3 ** (l + 1) - 1) // 2
            self.slice[l] = (start, end)

        # Register buffers for target shift and scale
        if target_shift is not None:
            self.target_shift = BufferDict({str(k): v for k, v in target_shift.items()})
        else:
            self.register_buffer("target_shift", None)
        if target_scale is not None:
            self.target_scale = BufferDict({str(k): v for k, v in target_scale.items()})
        else:
            self.register_buffer("target_scale", None)

        # Register a separate bias for each atomic type, only for rank-0 tensors.
        # Different for each feature dimension of the rank-0 tensor.
        if element_bias:
            self.element_bias = nn.Parameter(
                torch.zeros(num_atom_types, output_signature[0])
            )
        else:
            self.register_parameter("element_bias", None)

    def forward(self, atom_feats: list[Tensor], atom_type: Tensor) -> dict[int, Tensor]:
        """
        Args:
            atom_feats: list of atomic features, each of shape (n_atoms, F, T),
                where `F` is the channel dimension, and `T` is the tensor dimension.
            atom_type: The atomic type of each atom. Shape (n_atoms,).

        Returns:
            Dictionary of natural tensors for each atom. {l: T}, where `l` is the rank
            of the natural tensor, and `T` is the tensor. T has a shape of
            (n_atoms, n_l, 3**l), where `n_l` is the number of rank-l natural tensors.
            See `output_signature`.
        """
        assert len(atom_feats) == self.num_atom_feats, "Incorrect number of atom feats."

        # Select natural tensors of ranks needed for outputs
        # Each obtained by stacking the atomic features of all layers along the channel
        # dimension. {l: (n_atoms, F*self.num_atom_feats, 3 ** l)}
        atom_feats = {
            l: torch.cat([x[..., start:end] for x in atom_feats], dim=-2)
            for l, (start, end) in self.slice.items()
        }

        # Reshape the scalars from (..., F*num_atom_feats, 1) to (..., F*num_atom_feats)
        # This is needed by the MLP
        if 0 in atom_feats:
            atom_feats[0] = atom_feats[0].squeeze(-1)

        # Linear mapping F to n_l output dims for each atom; {l: (n_atoms, n_l, 3**l)}
        atom_out = {int(l): fn(atom_feats[int(l)]) for l, fn in self.kernel.items()}

        # Add the squeezed last dim back for scalars
        if 0 in atom_out:
            atom_out[0] = atom_out[0].unsqueeze(-1)

        # Normalization
        # Note, in a typical case, all ranks l will have a scale, but only rank-0 tensor
        # will have a shift. This is because the rank-0 tensor is a scalar, and we can
        # shift it to match the target.
        if self.target_scale is not None:
            for l, scale in self.target_scale.items():
                atom_out[int(l)] *= scale

        if self.target_shift is not None:
            for l, shift in self.target_shift.items():
                atom_out[int(l)] += shift

        # Learnable separate bias for each atom type (only for rank-0 tensors)
        if self.element_bias is not None:
            # Shape of atom_out[0]: (n_atoms, n_l, 1)
            atom_out[0] += self.element_bias[atom_type, :].unsqueeze(-1)

        return atom_out


class StructureTensor(nn.Module):
    """Get a tensor output for each configuration.

    For a model with multiple layers, the atomic feats of all layers are passed to this
    module. This module then
    1. selects the corresponding natural tensors,
    2. linearly maps them (via the channel dim) to get the atomic natural tensors,
    3. Pool (sum/mean) the atomic natural tensors to get the configuration tensor.


    Args:
        num_layers: Number of layers in the main model.
        in_features: Size of the input features, namely the channel dimension F.
        hidden_features: Size of the features for the hidden layers of MLP to process
            the scalar atom feats of the last layer. If a list, it provides the hidden
            layer sizes of the MLP. If an integer, it is interpreted as the number of
            hidden layers, and the hidden layer sizes are set to in_features.
            If the output has no rank-0 natural tensor, this argument is ignored.
        output_signature: A dictionary {l: n_l} that specifies the natural tensor
            components to output for each atomic configuration. The key `l` gives
            the rank of the natural tensor, and the value `n_l` gives the number of
            rank-l natural tensor to output. For example, a dielectric tensor is a
            symmetric rank-2 tensor, which can be decomposed as 1 rank-0  and 1 rank-2
            natural tensors. To model the dielectric tensor, the output_signature
            should be {0: 1, 1: 1, 2: 1}. As another example, the elastic tensor is a
            rank-4 tensor, which can be decomposed as 2 rank-0, 2 rank-2, and 1 rank-4
            natural tensors. To model the elastic tensor, the output_signature should
        num_atom_types: Number of atomic types.
        target_shift: A dictionary {l: shift} that specifies the shift to apply to
            the output of the model before computing the loss. Used together with
            target_scale.
        target_scale: A dictionary {l: scale} that specifies the scale to apply to
            the output of the model before computing the loss. Used together with
            target_shift. y = scale*z + shift, where z is the output of the
            network, and y is the predicted target.
            be {0: 2, 2: 2, 4: 1}.
        element_bias: If True, a separate bias is learned for each atomic type and
            added to the scalar part of the model.
        num_atom_feats: Number of atomic features to expect in the forward pass. If
            None, it is set to `num_layers`, indicating that the atomic features of
            all layers are passed to this module.
    """

    def __init__(
        self,
        num_layers: int,
        in_features: int,
        hidden_features: list[int] | int,
        output_signature: dict[int, int],
        num_atom_types: int,
        target_shift: dict[int, Tensor] = None,
        target_scale: dict[int, Tensor] = None,
        element_bias: bool = True,
        num_atom_feats: int = None,
    ):
        super().__init__()
        self.atomic_tensor_model = AtomicTensor(
            num_layers,
            in_features,
            hidden_features,
            output_signature,
            num_atom_types,
            target_shift,
            target_scale,
            element_bias,
            num_atom_feats,
        )

    def forward(
        self,
        atom_feats: list[Tensor],
        atom_type: Tensor,
        num_atoms: Tensor,
        reduce: str = "mean",
    ) -> dict[int, Tensor]:
        """
        Args:
            atom_feats: list of atomic features, each of shape (n_atoms, F, T),
                where `F` is the channel dimension, and `T` is the tensor dimension.
            atom_type: The atomic type of each atom. Shape (n_atoms,).
            num_atoms: The number of atoms in each atomic configuration.
                Shape (n_atoms,)
            reduce: Reduction method to use for pooling the atomic natural tensors to
                get the configuration tensor. Can be "mean" or "sum".

        Returns:
            Dictionary of natural tensors for each atomic configuration.
            {l: T}, where `l` is the rank of the natural tensor, and `T` is the tensor.
            T has a shape of (n_config, n_l, 3**l), where `n_l` is the number of rank-l
            natural tensors. See `output_signature`.
        """

        # Atomic tensor for each layer; {l: (n_atoms, n_l, 3**l)}
        atom_out = self.atomic_tensor_model(atom_feats, atom_type)

        # Gather to get output for each configuration; {l: (n_config, n_l, 3**l)
        conf_out = {
            l: scatter(x, torch.repeat_interleave(num_atoms), reduce=reduce, dim=0)
            for l, x in atom_out.items()
        }

        return conf_out
