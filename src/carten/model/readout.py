"""Readout layer."""

import torch
from torch import Tensor, nn

from carten.module.linear import LinearMap
from carten.module.mlp import MLP
from carten.module.normalize import LayerNorm
from carten.module.scatter import scatter


# TODO, this can be reimplemented using _AtomicScalar
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
        num_atom_types: Number of atomic types.
        atomic_shift: Shift of the output. See `atomic_scale`.
        atomic_scale: Scale of the output. The atomic shift and scale are used to
            transform the output. The final output for each atomic configuration is
            computed as y = x*scale + shift, where x result of the above four steps.
            If `atomic_shift` and `atomic_scale` is None, no transform is applied.
            If a scalar tensor is provided for `atomic_shift` and for `atomic_scale`,
            then it is then it is used for all atom types.
            If a tensor of shape (n_atom_types,) is provided for `atomic_shift` and
                `atomic_scale`, then it is applied to each atom type separately.
        element_bias: If True, a separate learnable bias is added for each atomic type.
    """

    def __init__(
        self,
        num_layers: int,
        in_features: int,
        hidden_features: list[int] | int,
        num_atom_types: int,
        atomic_shift: Tensor = None,
        atomic_scale: Tensor = None,
        element_bias: bool = True,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.num_atom_types = num_atom_types

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

        self.register_buffer("atomic_shift", atomic_shift)
        self.register_buffer("atomic_scale", atomic_scale)

        # Register a separate bias for each atomic type
        if element_bias:
            self.element_bias = nn.Parameter(torch.zeros(num_atom_types))
        else:
            self.register_parameter("element_bias", None)

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

        V = 0
        for i, fn in enumerate(self.out_layers):
            V += fn(atom_feats[i].squeeze(-1)).view(-1)  # shape (n_atoms,)

        # TODO, allow the per species scale and shift?
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

        # Bias for each atomic type
        if self.element_bias is not None:
            V += self.element_bias[atom_type]

        # Output of each configuration; (num_config,)
        out = scatter(V, torch.repeat_interleave(num_atoms), reduce="sum", dim=0)

        return out


class AtomicTensor(nn.Module):
    """Get a tensor output for each atom in a configuration.

    For a model with multiple layers, the atomic feats of all layers are passed to this
    module. This module then: linearly maps them (via the channel dim) to get the atomic
    natural tensors.

    Note, this module does not use `atomic_selector` to select a subset of atoms. If
    needed, it should be done before passing the atomic features to this module.

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
        use_layer_norm: bool = True,
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

        # Layer norm (nonlinearity) for the last layer
        if use_layer_norm:
            self.layer_norm = LayerNorm(
                dim=in_features,
                slice_sizes=[3**l for l in range(max(output_signature.keys()) + 1)],
            )
        else:
            self.register_buffer("layer_norm", None)

        self.kernel = nn.ModuleDict()
        self.slice = dict()

        for l, n in output_signature.items():
            start = (3**l - 1) // 2
            end = (3 ** (l + 1) - 1) // 2
            self.slice[l] = slice(start, end)

            if l == 0:
                layer = _AtomicScalar(
                    num_layers=num_layers,
                    in_features=in_features,
                    hidden_features=hidden_features,
                    out_features=n,
                    num_atom_types=num_atom_types,
                    atomic_shift=target_shift[l],
                    atomic_scale=target_scale[l],
                    element_bias=element_bias,
                    num_atom_feats=num_atom_feats,
                )
            else:
                layer = _AtomicTensor(
                    num_layers=num_layers,
                    in_features=in_features,
                    out_features=n,
                    num_atom_types=num_atom_types,
                    atomic_scale=target_scale[l],
                    num_atom_feats=num_atom_feats,
                )

            self.kernel[str(l)] = layer

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

        # Apply layer norm to the last layer's atomic feats
        if self.layer_norm is not None:
            atom_feats[-1] = self.layer_norm(atom_feats[-1])

        output = {}
        for l, s in self.slice.items():
            data = [x[..., s] for x in atom_feats]
            fn = self.kernel[str(l)]
            output[l] = fn(data, atom_type)

        return output


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
        use_layer_norm: bool = True,
    ):
        super().__init__()
        self.atomic_tensor_model = AtomicTensor(
            num_layers=num_layers,
            in_features=in_features,
            hidden_features=hidden_features,
            output_signature=output_signature,
            num_atom_types=num_atom_types,
            target_shift=target_shift,
            target_scale=target_scale,
            element_bias=element_bias,
            num_atom_feats=num_atom_feats,
            use_layer_norm=use_layer_norm,
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


class _AtomicScalar(nn.Module):
    def __init__(
        self,
        num_layers: int,
        in_features: int,
        hidden_features: list[int] | int,
        out_features: int,
        num_atom_types: int,
        atomic_shift: Tensor = None,
        atomic_scale: Tensor = None,
        element_bias: bool = True,
        num_atom_feats: int = None,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features
        self.num_atom_types = num_atom_types

        self.num_atom_feats = (
            num_atom_feats if num_atom_feats is not None else num_layers
        )

        self.layers = nn.ModuleList()

        for i in range(self.num_atom_feats):

            # Linear mapping for atom features in early layers
            if i < self.num_atom_feats - 1:
                self.layers.append(nn.Linear(in_features, out_features))

            # MLP for atom features in the last layer
            else:
                if isinstance(hidden_features, int):
                    hidden_features = [in_features for _ in range(hidden_features)]
                self.layers.append(
                    MLP(
                        in_features, out_features, hidden_features, out_activation=False
                    )
                )

        self.register_buffer("atomic_shift", atomic_shift)
        self.register_buffer("atomic_scale", atomic_scale)

        # Register a separate bias for each atomic type
        if element_bias:
            self.element_bias = nn.Parameter(torch.zeros(num_atom_types))
        else:
            self.register_parameter("element_bias", None)

    def forward(self, atom_feats: list[Tensor], atom_type: Tensor) -> Tensor:
        """
        Returns:
            Shape (n_atoms, out_features, 1).
        """
        assert len(atom_feats) == self.num_atom_feats, "Incorrect number of atom feats."

        V = 0
        for i, fn in enumerate(self.layers):
            # shape (n_atoms, out_features, 1)
            V += fn(atom_feats[i].squeeze(-1)).unsqueeze(-1)

        # TODO, allow the per species scale and shift?
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

        # Bias for each atomic type
        if self.element_bias is not None:
            V += self.element_bias[atom_type].view(-1, 1, 1)

        return V


class _AtomicTensor(nn.Module):
    def __init__(
        self,
        num_layers: int,
        in_features: int,
        out_features: int,
        num_atom_types: int,
        atomic_scale: Tensor = None,
        num_atom_feats: int = None,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.in_features = in_features
        self.out_features = out_features
        self.num_atom_types = num_atom_types

        self.num_atom_feats = (
            num_atom_feats if num_atom_feats is not None else num_layers
        )

        self.layers = nn.ModuleList()

        for i in range(self.num_atom_feats):

            # Linear mapping for atom features in early layers
            if i < self.num_atom_feats - 1:
                self.layers.append(LinearMap(in_features, out_features))

            # TODO, we can do a nonlinear layer and then this linear Map for this?
            # Linear mapping for atom features in the last layer
            else:
                self.layers.append(LinearMap(in_features, out_features))

        self.register_buffer("atomic_scale", atomic_scale)

    def forward(self, atom_feats: list[Tensor], atom_type: Tensor) -> Tensor:
        """
        Returns:
            Shape (n_atoms, out_features, 3**l).
        """
        assert len(atom_feats) == self.num_atom_feats, "Incorrect number of atom feats."

        V = 0
        for i, fn in enumerate(self.layers):
            V += fn(atom_feats[i])  # shape (n_atoms,out_features, 3**l)

        # TODO, allow the per species scale and shift?
        # Normalization
        if self.atomic_scale is not None:
            if self.atomic_scale.ndim == 0:
                V *= self.atomic_scale
            else:
                V *= self.atomic_scale[atom_type]

        return V
