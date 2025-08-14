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

        Z_i = atomic_number[i_idx]
        Z_j = atomic_number[j_idx]

        # a = (
        #     self.a_prefactor
        #     * 0.529
        #     / (torch.pow(Z_u, self.a_exp) + torch.pow(Z_v, self.a_exp))
        # )

        Z_pow = torch.pow(atomic_number, self.a_exp)
        a = self.a_prefactor * 0.529 / (Z_pow[i_idx] + Z_pow[j_idx])

        r = edge_vector.norm(p=2, dim=-1)
        r_over_a = r / a

        # phi = (
        #     self.c[0] * torch.exp(-3.2 * r_over_a)
        #     + self.c[1] * torch.exp(-0.9423 * r_over_a)
        #     + self.c[2] * torch.exp(-0.4028 * r_over_a)
        #     + self.c[3] * torch.exp(-0.2016 * r_over_a)
        # )

        # Optimized way to compute phi by calling exp once
        exponents = torch.stack(
            [
                -3.2 * r_over_a,
                -0.9423 * r_over_a,
                -0.4028 * r_over_a,
                -0.2016 * r_over_a,
            ],
            dim=-1,
        )  # Shape: (n_edges, 4)
        phi = torch.sum(self.c * torch.exp(exponents), dim=-1)

        v_edges = (14.3996 * Z_j * Z_i) / r * phi

        # Make it exactly zero outside r_max
        r_max = self.covalent_radii[Z_j] + self.covalent_radii[Z_i]
        envelope = dimenet_envelope(r / r_max, self.p)
        envelope.masked_fill_(r > r_max, 0.0)

        # 0.5: half to i and half to j
        v_edges = 0.5 * v_edges * envelope
        V_ZBL = scatter(v_edges, i_idx, dim=0, reduce="sum")

        return V_ZBL


@torch.jit.script
def zbl_energy(
    edge_vector: Tensor,
    edge_idx: Tensor,
    atomic_number: Tensor,
) -> torch.Tensor:
    """
    Compute ZBL energy of each atom.
    """
    dtype = edge_vector.dtype
    device = edge_vector.device

    # data from ase.data.covalent_radii
    # covalent_radii[1] is for H, covalent_radii[0] is for dummy element
    covalent_radii = torch.tensor(
        [
            0.2,
            0.31,
            0.28,
            1.28,
            0.96,
            0.84,
            0.76,
            0.71,
            0.66,
            0.57,
            0.58,
            1.66,
            1.41,
            1.21,
            1.11,
            1.07,
            1.05,
            1.02,
            1.06,
            2.03,
            1.76,
            1.7,
            1.6,
            1.53,
            1.39,
            1.39,
            1.32,
            1.26,
            1.24,
            1.32,
            1.22,
            1.22,
            1.2,
            1.19,
            1.2,
            1.2,
            1.16,
            2.2,
            1.95,
            1.9,
            1.75,
            1.64,
            1.54,
            1.47,
            1.46,
            1.42,
            1.39,
            1.45,
            1.44,
            1.42,
            1.39,
            1.39,
            1.38,
            1.39,
            1.4,
            2.44,
            2.15,
            2.07,
            2.04,
            2.03,
            2.01,
            1.99,
            1.98,
            1.98,
            1.96,
            1.94,
            1.92,
            1.92,
            1.89,
            1.9,
            1.87,
            1.87,
            1.75,
            1.7,
            1.62,
            1.51,
            1.44,
            1.41,
            1.36,
            1.36,
            1.32,
            1.45,
            1.46,
            1.48,
            1.4,
            1.5,
            1.5,
            2.6,
            2.21,
            2.15,
            2.06,
            2.0,
            1.96,
            1.9,
            1.87,
            1.8,
            1.69,
            0.2,
            0.2,
            0.2,
            0.2,
            0.2,
            0.2,
            0.2,
            0.2,
            0.2,
            0.2,
            0.2,
            0.2,
            0.2,
            0.2,
            0.2,
            0.2,
            0.2,
            0.2,
            0.2,
            0.2,
            0.2,
            0.2,
        ],
        dtype=dtype,
        device=device,
    )

    coeff = torch.tensor([0.1818, 0.5099, 0.2802, 0.02817], dtype=dtype, device=device)

    a_exp = 0.300
    a_prefactor = 0.4543
    p = 6

    # Indices of center atoms i and neighbor atoms j; (n_edges,)
    i_idx = edge_idx[0]
    j_idx = edge_idx[1]

    Z_i = atomic_number[i_idx]
    Z_j = atomic_number[j_idx]

    # a = (
    #     self.a_prefactor
    #     * 0.529
    #     / (torch.pow(Z_u, self.a_exp) + torch.pow(Z_v, self.a_exp))
    # )

    Z_pow = torch.pow(atomic_number, a_exp)
    a = a_prefactor * 0.529 / (Z_pow[i_idx] + Z_pow[j_idx])

    r = edge_vector.norm(p=2, dim=-1)
    r_over_a = r / a

    # phi = (
    #     self.c[0] * torch.exp(-3.2 * r_over_a)
    #     + self.c[1] * torch.exp(-0.9423 * r_over_a)
    #     + self.c[2] * torch.exp(-0.4028 * r_over_a)
    #     + self.c[3] * torch.exp(-0.2016 * r_over_a)
    # )

    # Optimized way to compute phi by calling exp once
    exponents = torch.stack(
        [
            -3.2 * r_over_a,
            -0.9423 * r_over_a,
            -0.4028 * r_over_a,
            -0.2016 * r_over_a,
        ],
        dim=-1,
    )  # Shape: (n_edges, 4)
    phi = torch.sum(coeff * torch.exp(exponents), dim=-1)

    v_edges = (14.3996 * Z_j * Z_i) / r * phi

    # Make it exactly zero outside r_max
    r_max = covalent_radii[Z_j] + covalent_radii[Z_i]
    envelope = dimenet_envelope(r / r_max, p)
    envelope.masked_fill_(r > r_max, 0.0)

    # 0.5: half to i and half to j
    v_edges = 0.5 * v_edges * envelope
    V_ZBL = scatter(v_edges, i_idx, dim=0, reduce="sum")

    return V_ZBL
