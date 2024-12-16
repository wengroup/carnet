"""Carten model to predict tensorial properties of materials and molecules."""
from torch import Tensor, nn

from .backbone import Backbone


class StructureTensorModel(nn.Module):
    """
    CARTEN model to predict a tensorial property for a material or molecular structure,
    such as dielectric and elastic tensors.
    """

    def __init__(self):
        super().__init__()


class AtomicTensorModel(nn.Module):
    """
    CARTEN model to predict a tensorial property for each atom in a system, such as
    NMR shielding tensors.
    """

    def __init__(self):
        super().__init__()
