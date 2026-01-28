import ase.data
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from carnet.module.radial import dimenet_envelope
from carnet.module.scatter import scatter


class ZBLEnergy(torch.nn.Module):
    """Ziegler-Biersack-Littmark (ZBL) potential with a polynomial cutoff envelope.

    Following the implementation in SchNetPack:
    https://github.com/atomistic-machine-learning/schnetpack/blob/30d86d9b17fe255ee7da72b84acf8bca1a0866fd/src/schnetpack/atomistic/nuclear_repulsion.py#L13


    Note, this is designed to work with units in eV and Angstroms.


    Args:
        p: order of the polynomial envelope (default: 6).
        trainable: whether the parameters `a_exp` and `a_prefactor` are trainable.
    """

    def __init__(self, p: int = 6, trainable: bool = True):
        super().__init__()

        dtype = torch.get_default_dtype()

        # Optionally trainable parameters
        # Since all quantities have a predefined sign, they are initialized to the
        # inverse softplus and a softplus function is applied in the forward pass to
        # guarantee the correct sign during training
        a0 = 0.529  # Bohr radius in Angstroms
        a_prefactor = inverse_softplus(torch.tensor(0.8854 * a0, dtype=dtype))
        a_pow = inverse_softplus(torch.tensor(0.23, dtype=dtype))

        self.a_prefactor = nn.Parameter(a_prefactor, requires_grad=trainable)
        self.a_pow = nn.Parameter(a_pow, requires_grad=trainable)

        # Do not make these trainable
        exponents = torch.tensor([3.19980, 0.94229, 0.40290, 0.20162], dtype=dtype)
        coefficients = torch.tensor([0.18175, 0.50986, 0.28022, 0.02817], dtype=dtype)
        # e^2 / (4 * pi * epsilon_0) in eV * Angstrom
        factor = torch.tensor(14.3996, dtype=dtype)

        self.exponents = nn.Parameter(exponents, requires_grad=False)
        self.coefficients = nn.Parameter(coefficients, requires_grad=False)
        self.factor = nn.Parameter(factor, requires_grad=False)

        # Other constants
        self.register_buffer("p", torch.tensor(p, dtype=torch.int))
        self.register_buffer(
            "covalent_radii", torch.tensor(ase.data.covalent_radii, dtype=dtype)
        )

    def forward(
        self,
        edge_vector: Tensor,
        edge_idx: Tensor,
        atomic_number: Tensor,
    ) -> torch.Tensor:
        """
        Compute ZBL energy of each atom.
        """

        # Indices of center atoms i and neighbor atoms j; (n_edges,)
        i_idx = edge_idx[0]
        j_idx = edge_idx[1]
        Z = atomic_number

        Z_pow = Z ** F.softplus(self.a_pow)
        a = F.softplus(self.a_prefactor) / (Z_pow[i_idx] + Z_pow[j_idx])

        r = edge_vector.norm(p=2, dim=-1)
        r_over_a = r / a

        phi = torch.sum(
            self.coefficients * torch.exp(-self.exponents * r_over_a.unsqueeze(-1)),
            dim=-1,
        )

        v_edges = self.factor * Z[i_idx] * Z[j_idx] * phi / r

        # Make it exactly zero outside r_max
        r_max = self.covalent_radii[Z[i_idx]] + self.covalent_radii[Z[j_idx]]
        envelope = dimenet_envelope(r / r_max, self.p)
        envelope.masked_fill_(r > r_max, 0.0)

        # 0.5: half to i and half to j
        v_edges = 0.5 * v_edges * envelope
        v_atom = scatter(v_edges, i_idx, dim=0, reduce="sum", dim_size=Z.shape[0])

        return v_atom


def inverse_softplus(x: Tensor):
    return torch.log(torch.exp(x) - 1.0)
