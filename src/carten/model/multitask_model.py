"""Carten model to predict tensorial properties of materials and molecules."""

from torch import Tensor, nn

from .backbone import Backbone
from .readout import AtomicTensor, StructureScalar, StructureTensor


class MultiTaskModel(nn.Module):
    """
    CARTEN model to predict a tensorial property for a material or molecular structure,
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
        max_chebyshev_degree: int = 8,
        radial_mlp_hidden_layers: list[int] | int = 2,
        atomic_moment_mode: str = "vanilla",
        tp_path_mode: str = "full",
        level: int = None,
        #
        layer_norm: bool = True,
        activation: str = "silu",
        last_layer_activation: bool = False,
        residual: bool = True,
        # optional layers
        use_linear_channel_input: bool = False,
        use_linear_channel_residual: bool = True,
        # output
        target_name: list[str] = None,
        target_shift: dict[str, dict[str, Tensor]] = None,
        target_scale: dict[str, dict[str, Tensor]] = None,
        output_mlp_hidden_layers: list[int] | int = 2,
        output_signature: dict[int, int] = None,
        output_from_all_layers: bool = False,
        element_bias: bool = True,
        use_layer_norm: bool = True,
        use_atomic_dependent_weight: bool = True,
    ):
        super().__init__()

        self.output_from_all_layers = output_from_all_layers
        if output_from_all_layers:
            num_atom_feats = num_layers
        else:
            num_atom_feats = 1

        max_out_L = max([max(d.keys()) for k, d in output_signature.items()])

        self.backbone = Backbone(
            F=F,
            max_L=max_L,
            num_atom_types=num_atom_types,
            r_cut=r_cut,
            num_layers=num_layers,
            num_average_neigh=num_average_neigh,
            max_out_L=max_out_L,
            max_degree=max_degree,
            max_chebyshev_degree=max_chebyshev_degree,
            radial_mlp_hidden_layers=radial_mlp_hidden_layers,
            atomic_moment_mode=atomic_moment_mode,
            tp_path_mode=tp_path_mode,
            level=level,
            layer_norm=layer_norm,
            activation=activation,
            last_layer_activation=last_layer_activation,
            residual=residual,
            use_linear_channel_input=use_linear_channel_input,
            use_linear_channel_residual=use_linear_channel_residual,
            use_atomic_dependent_weight=use_atomic_dependent_weight,
        )

        # There will be multiple output head

        self.output_heads = nn.ModuleDict()

        if "energy" in target_name or "forces" in target_name:
            self.output_heads["energy"] = StructureScalar(
                num_layers=num_layers,
                in_features=F,
                hidden_features=output_mlp_hidden_layers,
                num_atom_types=num_atom_types,
                atomic_shift=target_shift["energy"],
                atomic_scale=target_scale["energy"],
                element_bias=element_bias,
            )

        for name in target_name:
            if name in ["energy", "forces"]:
                pass
            # Structure tensor
            elif name in ["polarizability_tensor", "dipole_moment_tensor"]:
                self.output_heads[name] = StructureTensor(
                    num_layers=num_layers,
                    in_features=F,
                    hidden_features=output_mlp_hidden_layers,
                    output_signature=output_signature[name],
                    num_atom_types=num_atom_types,
                    # TODO, the _natural is hard-coded
                    target_shift=target_shift[name + "_natural"],
                    target_scale=target_scale[name + "_natural"],
                    element_bias=element_bias,
                    num_atom_feats=num_atom_feats,
                    reduce="sum",
                    use_layer_norm=use_layer_norm,
                )
            # Atomic tensor
            elif name in ["shielding_tensor"]:
                self.output_heads[name] = AtomicTensor(
                    num_layers=num_layers,
                    in_features=F,
                    hidden_features=output_mlp_hidden_layers,
                    output_signature=output_signature[name],
                    num_atom_types=num_atom_types,
                    # TODO, the _natural is hard-coded
                    target_shift=target_shift[name + "_natural"],
                    target_scale=target_scale[name + "_natural"],
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
    ) -> dict[str, Tensor]:
        # Get the atom feats
        all_atom_feats = self.backbone(
            edge_vector,
            edge_idx,
            atom_type,
            num_atoms,
            return_all=self.output_from_all_layers,
            scalar_only=False,
        )

        output = {}
        for target_name, head in self.output_heads.items():
            if target_name == "energy":
                # Energy, only needs the scalar features
                selected_feats = [feats[..., 0:1] for feats in all_atom_feats]
                output[target_name] = head(selected_feats, atom_type, num_atoms)
            elif target_name == "dipole_moment_tensor":
                selected_feats = all_atom_feats
                output[target_name] = head(selected_feats, atom_type, num_atoms)
            elif target_name == "polarizability_tensor":
                selected_feats = all_atom_feats
                output[target_name] = head(selected_feats, atom_type, num_atoms)
            elif target_name == "shielding_tensor":
                selected_feats = all_atom_feats
                output[target_name] = head(selected_feats, atom_type)
            else:
                raise ValueError(f"Unknown target mode: {target_name}.")

        return output
