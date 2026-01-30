"""Radial basis functions."""

import torch
from torch import Tensor, nn


class RadialBasis(nn.Module):
    r"""Radial basis functions.

    Weight: not depend on chemical species.
    output_dim: radial_basis_degree + 1 if chebyshev, radial_basis_degree if bessel.
    """

    def __init__(
        self,
        radial_basis_degree: int = 8,
        r_cut: float = 5,
        envelope: int = 6,
        basis_type: str = "bessel",
    ):
        """
        Args:
            radial_basis_degree: max degree of the Chebyshev polynomial or number
                of basis functions. The total number of functions is
                `radial_basis_degree + 1` for chebyshev and `radial_basis_degree` for bessel.
            r_cut: cutoff distance.
            envelope: power of the dimenet_envelope function.
            basis_type: Type of basis function to use. Either "chebyshev" or "bessel".
        """
        super().__init__()
        self.radial_basis_degree = radial_basis_degree
        self.r_cut = r_cut
        self.envelope = envelope
        self.basis_type = basis_type

        if basis_type == "chebyshev":
            output_dim = radial_basis_degree + 1
            self.basis = ChebyshevBasis(degree=radial_basis_degree, r_cut=r_cut)
        elif basis_type == "bessel":
            output_dim = radial_basis_degree
            self.basis = BesselBasis(r_cut=r_cut, num_basis=radial_basis_degree)
        else:
            raise ValueError(f"Unknown basis type: {basis_type}")

        self.register_buffer("output_dim", torch.tensor(output_dim, dtype=torch.int))

    def forward(self, r: Tensor):
        """
        Args:
            r: 1D tensor of distances between atoms i and j.

        Returns:
            A tensor of shape (len(r), output_dim).
        """
        basis_out = self.basis(r)
        env = dimenet_envelope(r, self.r_cut, p=self.envelope)

        return basis_out * env.unsqueeze(-1)


class ChebyshevBasis(nn.Module):
    """
    Chebyshev polynomial basis functions.
    """

    def __init__(self, degree: int, r_cut: float):
        super().__init__()
        self.degree = degree
        self.register_buffer("r_cut", torch.tensor(r_cut))

    def forward(self, r: Tensor) -> Tensor:
        # Normalize distance to [0, 1] range
        x = (r / self.r_cut).clamp(0.0, 1.0)

        # Chebyshev polynomials of the first kind.
        # Note: Should not use torch.special.chebyshev_polynomial_t, because it does
        # NOT support gradient calculation (upto torch v-2.8.1).

        T = [torch.ones_like(x), x]  # T0 and T1
        for i in range(2, self.degree + 1):
            T.append(2.0 * x * T[i - 1] - T[i - 2])

        return torch.stack(T, dim=-1)


class BesselBasis(nn.Module):
    """
    Bessel basis functions using the 0th spherical Bessel function j0.
    For more information, see the DimeNet paper: https://arxiv.org/abs/2003.03123
    """

    def __init__(self, r_cut: float, num_basis: int = 8):
        super().__init__()
        self.num_basis = num_basis

        self.register_buffer("freqs", torch.arange(1, num_basis + 1) * torch.pi / r_cut)
        self.register_buffer("prefactor", torch.tensor((2.0 / r_cut) ** 0.5))

    def forward(self, r: Tensor) -> Tensor:
        return (
            self.prefactor * torch.sin(self.freqs * r.unsqueeze(-1)) / r.unsqueeze(-1)
        )


@torch.jit.script
def dimenet_envelope(r: Tensor, r_cut: float, p: int = 6):
    """The envelope function used in DimNet.

    1 - (p+1)(p+2)/2*x**p + p*(p+2)*x**(p+1) - p*(p+1)/2*x**(p+2)

    Args:
        r: distance tensor.
        r_cut: cutoff distance.
        p: power.
    """
    x = (r / r_cut).clamp(0.0, 1.0)

    if p == 6:
        env = 1 - 28 * x**6 + 48 * x**7 - 21 * x**8
    elif p == 3:
        env = 1 - 10 * x**3 + 15 * x**4 - 6 * x**5
    else:
        env = (
            1
            - (p + 1) * (p + 2) / 2 * x**p
            + p * (p + 2) * x ** (p + 1)
            - p * (p + 1) / 2 * x ** (p + 2)
        )

    return env
