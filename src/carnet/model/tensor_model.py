"""Model forpredicting tensorial properties of materials and molecules."""

from torch import Tensor, nn

from .backbone import Backbone
from .readout import AtomicTensor, StructureTensor


class AtomicTensorModel(nn.Module):
    """
    Model to predict a tensorial property for each atom in a system, such as
    NMR shielding tensors.
    """

    def __init__(
        self,
        F: int,
        max_L: int,
        num_atom_types: int,
        r_cut: float,
        num_layers: int,
        num_average_neigh: float,
        # angular
        max_degree: int = 3,
        # radial
        radial_basis_degree: int = 8,
        radial_basis_type: str = "bessel",
        radial_mlp_hidden_layers: list[int] | int = 2,
        #
        tp_path_mode: str = "full",
        tp_path_polar_only: bool = False,
        level: int = None,
        # optional layers
        use_linear_channel_input: bool = False,
        use_linear_channel_hyper: bool = False,
        use_linear_channel_residual: bool = True,
        use_atomic_dependent_weight: bool = True,
        residual: bool = True,
        # output
        output_mlp_hidden_layers: list[int] | int = 2,
        target_shift: dict[str, Tensor] = None,
        target_scale: dict[str, Tensor] = None,
        output_signature: dict[int, int] = None,
        output_from_all_layers: bool = False,
        element_bias: bool = True,
        use_layer_norm: bool = True,  # layer normalization for the readout
    ):
        """
        Args:
            target_shift: A dictionary {l: shift} that specifies the shift to apply to
                the output of the model before computing the loss. Used together with
                target_scale.
            target_scale: A dictionary {l: scale} that specifies the scale to apply to
                the output of the model before computing the loss. Used together with
                target_shift. y = scale*z + shift, where z is the output of the
                network, and y is the predicted target.
            output_signature: A dictionary {l: n_l} that specifies the natural tensor
                components to output for each atomic configuration. The key `l` gives
                the rank of the natural tensor, and the value `n_l` gives the number of
                rank-l natural tensor to output. For example, a dielectric tensor is a
                symmetric rank-2 tensor, which can be decomposed as 1 rank-0  and 1
                rank-2 natural tensors. To model the dielectric tensor, the
                output_signature should be {0: 1, 1: 1, 2: 1}. As another example,
                the elastic tensor is a rank-4 tensor, which can be decomposed as 2
                rank-0, 2 rank-2, and 1 rank-4 natural tensors. To model the elastic
                tensor, the output_signature should be {0: 2, 2: 2, 4: 1}.
            output_from_all_layers: If True, the output is constructed from the
                atom features of all layers. If False, the output is constructed from
                the atom features of the last layer only.
            element_bias: If True, a separate bias is learned for each atomic type and
            added to the scalar part of the model.
        """
        super().__init__()
        self.output_from_all_layers = output_from_all_layers

        if output_from_all_layers:
            num_atom_feats = num_layers
        else:
            num_atom_feats = 1

        self.backbone = Backbone(
            F=F,
            max_L=max_L,
            num_atom_types=num_atom_types,
            r_cut=r_cut,
            num_layers=num_layers,
            num_average_neigh=num_average_neigh,
            max_out_L=max(output_signature.keys()),
            max_degree=max_degree,
            radial_basis_degree=radial_basis_degree,
            radial_basis_type=radial_basis_type,
            radial_mlp_hidden_layers=radial_mlp_hidden_layers,
            tp_path_mode=tp_path_mode,
            tp_path_polar_only=tp_path_polar_only,
            level=level,
            use_linear_channel_input=use_linear_channel_input,
            use_linear_channel_hyper=use_linear_channel_hyper,
            use_linear_channel_residual=use_linear_channel_residual,
            use_atomic_dependent_weight=use_atomic_dependent_weight,
            residual=residual,
        )

        self.readout = AtomicTensor(
            num_layers=num_layers,
            in_features=F,
            hidden_features=output_mlp_hidden_layers,
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
        edge_vector: Tensor,
        edge_idx: Tensor,
        atom_type: Tensor,
        num_atoms: Tensor,
        atomic_selector: list[bool] = None,
    ) -> Tensor:
        """
        Args:
            edge_vector:
            edge_idx:
            atom_type:
            num_atoms:
            atomic_selector: A list of boolean values that indicates the features of which
                atoms to use for later processing. The length of the list should be equal to
                the number of atoms. If None, the features of all atoms are used.
        """

        # Get the atom feats
        all_atom_feats = self.backbone(
            edge_vector,
            edge_idx,
            atom_type,
            num_atoms,
            return_all=self.output_from_all_layers,
            scalar_only=False,
        )

        # Select the atomic features of the specified atoms
        if atomic_selector is not None:
            all_atom_feats = [x[atomic_selector] for x in all_atom_feats]
            atom_type = atom_type[atomic_selector]

        # Compute the atomic tensor on each atom
        output = self.readout(all_atom_feats, atom_type)

        return output


class StructureTensorModel(nn.Module):
    """
    CarNet model to predict a tensorial property for a material or molecular structure,
    such as dielectric and elastic tensors.
    """

    def __init__(
        self,
        F: int,
        max_L: int,
        num_atom_types: int,
        r_cut: float,
        num_layers: int,
        num_average_neigh: float,
        # angular
        max_degree: int = 3,
        # radial
        radial_basis_degree: int = 8,
        radial_basis_type: str = "bessel",
        radial_mlp_hidden_layers: list[int] | int = 2,
        #
        tp_path_mode: str = "full",
        tp_path_polar_only: bool = False,
        level: int = None,
        # optional layers
        use_linear_channel_input: bool = False,
        use_linear_channel_hyper: bool = False,
        use_linear_channel_residual: bool = True,
        use_atomic_dependent_weight: bool = True,
        residual: bool = True,
        # output
        output_mlp_hidden_layers: list[int] | int = 2,
        target_shift: dict[str, Tensor] = None,
        target_scale: dict[str, Tensor] = None,
        output_signature: dict[int, int] = None,
        output_from_all_layers: bool = False,
        reduce: str = "mean",
        element_bias: bool = True,
        use_layer_norm: bool = True,  # layer normalization for the readout
    ):
        super().__init__()

        self.output_from_all_layers = output_from_all_layers
        if output_from_all_layers:
            num_atom_feats = num_layers
        else:
            num_atom_feats = 1

        self.backbone = Backbone(
            F=F,
            max_L=max_L,
            num_atom_types=num_atom_types,
            r_cut=r_cut,
            num_layers=num_layers,
            num_average_neigh=num_average_neigh,
            max_out_L=max(output_signature.keys()),
            max_degree=max_degree,
            radial_basis_degree=radial_basis_degree,
            radial_basis_type=radial_basis_type,
            radial_mlp_hidden_layers=radial_mlp_hidden_layers,
            tp_path_mode=tp_path_mode,
            tp_path_polar_only=tp_path_polar_only,
            level=level,
            use_linear_channel_input=use_linear_channel_input,
            use_linear_channel_hyper=use_linear_channel_hyper,
            use_linear_channel_residual=use_linear_channel_residual,
            use_atomic_dependent_weight=use_atomic_dependent_weight,
            residual=residual,
        )

        self.readout = StructureTensor(
            num_layers=num_layers,
            in_features=F,
            hidden_features=output_mlp_hidden_layers,
            output_signature=output_signature,
            num_atom_types=num_atom_types,
            target_shift=target_shift,
            target_scale=target_scale,
            element_bias=element_bias,
            num_atom_feats=num_atom_feats,
            use_layer_norm=use_layer_norm,
            reduce=reduce,
        )

    def forward(
        self,
        edge_vector: Tensor,
        edge_idx: Tensor,
        atom_type: Tensor,
        num_atoms: Tensor,
        atomic_selector: list[bool] = None,
        # `atomic_selector` is not needed; it is added for API uniformity with the
        # AtomicTensorModel()
        batch: Tensor = None,
    ) -> Tensor:
        """
        Args:
            edge_vector:
            edge_idx:
            atom_type:
            num_atoms:
            atomic_selector: A list of boolean values that indicates the features of which
                atoms to use for later processing. The length of the list should be equal to
                the number of atoms. If None, the features of all atoms are used.
            batch: Tensor of shape (n_atoms,) that identifies the configuration each
                atom belongs to. If None, it is inferred from `num_atoms`.
        """
        # Get the atom feats
        all_atom_feats = self.backbone(
            edge_vector,
            edge_idx,
            atom_type,
            num_atoms,
            return_all=self.output_from_all_layers,
            scalar_only=False,
        )

        # Compute the structure tensor of each configuration
        output = self.readout(all_atom_feats, atom_type, num_atoms, batch)

        return output
