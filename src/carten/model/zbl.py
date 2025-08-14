import ase.data
import torch
from torch import Tensor

from carten.module.radial import dimenet_envelope
from carten.module.scatter import scatter


class ZBL(torch.nn.Module):
    """Ziegler-Biersack-Littmark (ZBL) potential with a polynomial cutoff envelope.

    Following the implementation in MACE:
    https://github.com/ACEsuit/mace/blob/9d31ac2c86ebc88c7a843fa7a3dfe360b276f08b/mace/modules/radial.py#L147C1-L219C1
    """

    p: torch.Tensor

    def __init__(self, p: int = 6, trainable: bool = False):
        super().__init__()

        dtype = torch.get_default_dtype()

        # Pre-calculate the p coefficients for the ZBL potential
        self.register_buffer(
            "c",
            torch.tensor([0.1818, 0.5099, 0.2802, 0.02817], dtype=dtype),
        )
        self.register_buffer("p", torch.tensor(p, dtype=torch.int))
        self.register_buffer(
            "covalent_radii", torch.tensor(ase.data.covalent_radii, dtype=dtype)
        )

        if trainable:
            self.a_exp = torch.nn.Parameter(torch.tensor(0.300, dtype=dtype))
            self.a_prefactor = torch.nn.Parameter(torch.tensor(0.4543, dtype=dtype))
        else:
            self.register_buffer("a_exp", torch.tensor(0.300, dtype=dtype))
            self.register_buffer("a_prefactor", torch.tensor(0.4543, dtype=dtype))

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

        Z_u = atomic_number[j_idx]
        Z_v = atomic_number[i_idx]

        a = (
            self.a_prefactor
            * 0.529
            / (torch.pow(Z_u, self.a_exp) + torch.pow(Z_v, self.a_exp))
        )

        r = edge_vector.norm(p=2, dim=-1)
        r_over_a = r / a

        phi = (
            self.c[0] * torch.exp(-3.2 * r_over_a)
            + self.c[1] * torch.exp(-0.9423 * r_over_a)
            + self.c[2] * torch.exp(-0.4028 * r_over_a)
            + self.c[3] * torch.exp(-0.2016 * r_over_a)
        )
        v_edges = (14.3996 * Z_u * Z_v) / r * phi

        # TODO, in fact, it is not needed to have an envelope, directly using v_edges
        #  should be fine
        # Make it exactly zero outside r_max
        r_max = self.covalent_radii[Z_u] + self.covalent_radii[Z_v]
        envelope = dimenet_envelope(r / r_max, self.p)

        # r can be larger than r_max; this ensures the envelope is zero outside r_max
        envelope = envelope * (r < r_max)

        # 0.5: half to i and half to j
        v_edges = 0.5 * v_edges * envelope
        V_ZBL = scatter(v_edges, i_idx, dim=0, reduce="sum")

        return V_ZBL
