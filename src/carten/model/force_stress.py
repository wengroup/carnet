import torch
from torch import Tensor


def compute_forces(energy: Tensor, coords: Tensor, training: bool = False) -> Tensor:
    """Compute the forces on atoms.

    The forces are computed as the negative gradient of the energy with respect to the
    coordinates of the atoms.

    This works for a batch of configurations.

    Args:
        energy: shape(n_configs,) total energy of the configurations.
        coords: shape(n_atoms, 3), coordinates of the atoms. This is the batched coords
            of all atoms in all configurations.
        training: Whether the model is in training mode.

    Returns:
        Forces on the atoms. Shape (n_atoms, 3).
    """

    # Note: autograd.grad computes the sum of the gradients of energy with respect
    # to the coords. This is OK because the energy of each configuration is independent
    # of the coords of other configurations, and thus the gradients of configuration j
    # to the coords of configuration i is 0.
    # TODO, check the `is_grad_batched` argument of autograd.grad to see if we can do
    #  batched gradient computation.
    return -torch.autograd.grad(energy.sum(), coords, create_graph=training)[0]


def compute_forces_stress(
    energy: Tensor, coords: Tensor, cell: Tensor, strain: Tensor, training: bool = False
) -> tuple[Tensor, Tensor]:
    """Compute the virial stress on atoms.

    The virial stress is computed as the gradient of the energy with respect to the cell
    vectors.

    This works for a batch of configurations.

    Args:
        energy: shape(n_configs,) total energy of the configurations.
        coords: shape(n_atoms, 3), coordinates of the atoms. This is the batched coords
            of all atoms in all configurations.
        cell: shape(B, 3, 3), unit cell. B is the batch size.
        strain: shape(B, 3, 3), strain tensor. B is the batch size.
        training: Whether the model is in training mode.

    Return:
        Forces on the atoms. Shape (n_atoms, 3).
        Virial stress on the atoms. Shape (B, 3, 3).
    """
    grad = torch.autograd.grad([energy.sum()], [coords, strain], create_graph=training)

    forces = -grad[0]  # (n_atoms, 3)

    # det is equal to a dot (b cross c)
    # [B, 3, 3] -> [B]
    volume = torch.linalg.det(cell).abs()
    viral = grad[1]  # (B, 3, 3)
    stress = viral / volume.view(-1, 1, 1)

    return forces, stress


def compute_forces_stress_single_config(
    energy: Tensor, coords: Tensor, cell: Tensor, strain: Tensor, training: bool = False
) -> tuple[Tensor, Tensor]:
    """Compute the virial stress on atoms.

    The virial stress is computed as the gradient of the energy with respect to the cell
    vectors.

    This works for a single configuration.

    Args:
        energy: shape(n_configs,) total energy of the configurations.
        coords: shape(n_atoms, 3), coordinates of the atoms. This is the batched coords
            of all atoms in all configurations.
        cell: shape(3, 3), unit cell.
        strain: shape(3, 3), strain tensor.
        training: Whether the model is in training mode.

    Return:
        Forces on the atoms. Shape (n_atoms, 3).
        Virial stress on the atoms. Shape (3, 3).
    """
    grad = torch.autograd.grad([energy.sum()], [coords, strain], create_graph=training)

    forces = -grad[0]  # (n_atoms, 3)

    # det is equal to a dot (b cross c)
    volume = torch.linalg.det(cell).abs()
    viral = grad[1]  # (3, 3)
    stress = viral / volume

    return forces, stress


def apply_strain(
    pos: Tensor, cell: Tensor, batch: Tensor
) -> tuple[Tensor, Tensor, Tensor]:
    """
    This is required to calculate the stress as a response property.
    Adds strain-dependence to absolute positions and unit cell.

    This works for a batch of configurations.

    This is based on schnetpack and nequip, see
    https://github.com/atomistic-machine-learning/schnetpack/blob/643c9a1ab17757b5fd1f94dabec50fb72dbde025/src/schnetpack/atomistic/response.py#L434
    https://github.com/mir-group/nequip/blob/d3a7763228baf89c28506098d06254011440fc9b/nequip/nn/_grad_output.py#L177

    This is based on the below paper:
    Knuth et. al. Comput. Phys. Commun 190, 33-50, 2015
    https://pure.mpg.de/rest/items/item_2085135_9/component/file_2156800/content

    Args:
        pos: shape(n_atoms, 3), absolute positions of the atoms, batch of atoms from all
            configurations.
        cell: shape(B, 3, 3), unit cell. B, batch size.
        batch: shape(n_atoms,), batch index of each atom.

    Returns:
        strain: shape(B, 3, 3), strain tensor.
        strained_pos: shape(n_atoms, 3), strained positions of the atoms.
        strained_cell: shape(B, 3, 3), strained unit cell.
    """

    strain = torch.zeros_like(cell)  # (B, 3, 3)
    strain.requires_grad_(True)

    # in the above paper, the infinitesimal distortion is *symmetric*
    # so we symmetrize the displacement before applying it to
    # the positions/cell
    strain_sym = 0.5 * (strain + strain.transpose(-1, -2))

    # strain positions
    # batched [n_atoms, 1, 3] @ [n_atoms, 3, 3] -> [n_atoms, 1, 3] -> [n_atoms, 3]
    strained_pos = pos + torch.bmm(pos.unsqueeze(-2), strain_sym[batch]).squeeze(1)

    # strain cell
    # [B, 3, 3] @ [B, 3, 3] -> [B, 3, 3]
    strained_cell = cell + torch.bmm(cell, strain_sym)

    return strain, strained_pos, strained_cell


def apply_strain_single_config(
    pos: Tensor, cell: Tensor
) -> tuple[Tensor, Tensor, Tensor]:
    """
    This is required to calculate the stress as a response property.
    Adds strain-dependence to absolute positions and unit cell.

    This works for a single configuration.

    Args:
        pos: shape(n_atoms, 3), absolute positions of the atoms.
        cell: shape(3, 3), unit cell.

    Returns:
        strain: shape(3, 3), strain tensor.
        strained_pos: shape(n_atoms, 3), strained positions of the atoms.
        strained_cell: shape(3, 3), strained unit cell.
    """

    strain = torch.zeros_like(cell)  # (3, 3)
    strain.requires_grad_(True)

    strain_sym = 0.5 * (strain + strain.transpose(0, 1))

    # strain positions
    strained_pos = torch.addmm(pos, pos, strain_sym)

    # strain cell
    # [3, 3] @ [3, 3] -> [3, 3]
    strained_cell = torch.addmm(cell, cell, strain_sym)

    return strain, strained_pos, strained_cell
