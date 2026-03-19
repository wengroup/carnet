from typing import Any, Optional, Tuple

import torch


# pylint: disable=abstract-method, arguments-differ
class LAMMPS_Exchange(torch.autograd.Function):
    """
    Autograd function for MPI communication in LAMMPS MLIAP.
    This allows the model to synchronize features between local and ghost atoms
    during the forward pass, and propagate gradients during the backward pass.
    """

    @staticmethod
    def forward(ctx, feats: torch.Tensor, lammps_class: Any) -> torch.Tensor:
        ctx.vec_len = feats.shape[-1]
        ctx.lammps_class = lammps_class
        out = torch.empty_like(feats)
        lammps_class.forward_exchange(feats, out, ctx.vec_len)
        return out

    @staticmethod
    def backward(ctx, *grad_outputs):
        (grad,) = grad_outputs
        gout = torch.empty_like(grad)
        ctx.lammps_class.reverse_exchange(grad, gout, ctx.vec_len)
        return gout, None


def handle_lammps(
    node_feats: torch.Tensor,
    lammps_class: Optional[Any],
    lammps_natoms: Tuple[int, int],
    first_layer: bool,
) -> torch.Tensor:
    """
    Handle feature exchange for LAMMPS MLIAP.
    If running in LAMMPS and it's not the first layer, synchronize features
    between local and ghost atoms to expand the receptive field.

    Args:
        node_feats: Atomic features. Shape (n_atoms, F, T).
        lammps_class: The LAMMPS MLIAP Python object (the 'data' object from LAMMPS),
            which provides forward_exchange and reverse_exchange methods.
        lammps_natoms: A tuple of (n_local, n_ghost) atoms.
        first_layer: Whether this is the first interaction layer. If True,
            exchange is skipped as the receptive field is initially just the
            single-layer cutoff.
    """
    if lammps_class is None or first_layer or torch.jit.is_scripting():
        return node_feats

    node_feats = node_feats.contiguous()
    n_real, n_ghost = lammps_natoms
    expected_total = n_real + n_ghost

    # If input already includes ghost slots, skip padding but still do exchange.
    if node_feats.shape[0] == expected_total:
        node_feats = LAMMPS_Exchange.apply(node_feats, lammps_class)
        return node_feats

    # Normal case: pad with zeros for ghosts, then exchange
    pad_shape = (n_ghost,) + node_feats.shape[1:]
    pad = torch.zeros(
        pad_shape,
        dtype=node_feats.dtype,
        device=node_feats.device,
    )
    node_feats = torch.cat((node_feats, pad), dim=0)
    node_feats = LAMMPS_Exchange.apply(node_feats, lammps_class)
    return node_feats


def truncate_ghosts(tensor: torch.Tensor, n_real: Optional[int] = None) -> torch.Tensor:
    """Truncate the tensor to only keep the real atoms."""
    return tensor[:n_real] if n_real is not None else tensor
