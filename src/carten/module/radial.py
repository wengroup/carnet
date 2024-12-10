"""Radial basis functions."""


from typing import Optional

import torch
from torch import Tensor
from torch import nn as nn


class RadialPart(nn.Module):
    """Radial part of the MTP.

    f_nu_i_j(r) = \sum_\beta c_nu_i_j * radial_basis_\beta(r)

    Eq. 3 of Shapeev.
    """

    def __init__(
        self,
        n_u: int,
        n_z: int,
        max_chebyshev_degree: int = 9,
        r_cut: float = 5,
        envelope: Optional[int] = None,
    ):
        """
        Args:
            n_u: number of radial basis functions.
            n_z: number of atom types.
            max_chebyshev_degree: max degree of the Chebyshev polynomial. The total
                number of chebyshev polynomials is `max_chebyshev_degree + 1`; +1 for
                the zeroth degree.
            r_cut: cutoff distance.
            envelope: envelope function to make the radial basis function smooth at
                r_cut. if None, using the MTP 2nd order polynomial envelope. Otherwise,
                p is a positive integer, and the envelope function in dimenet is used.
        """
        super().__init__()

        self.n_u = n_u
        self.n_z = n_z
        self.max_chebyshev_degree = max_chebyshev_degree
        self.r_cut = r_cut
        self.envelope = envelope

        self.c = nn.Parameter(torch.empty(n_z, n_z, n_u, max_chebyshev_degree + 1))
        self.reset_parameters()

    def reset_parameters(self):
        """Initialize the weights to:

            uniform(-1/sqrt(in_features), 1/sqrt(in_features)).

        Note, self.c can be regarded as a collection of multiple linear layers, each for
        a specific combination of zi and zj.

        https://github.com/pytorch/pytorch/blob/e3ca7346ce37d756903c06e69850bdff135b6009/torch/nn/modules/linear.py#L109
        """
        k = 1 / (self.max_chebyshev_degree + 1) ** 0.5
        nn.init.uniform_(self.c, -k, k)

    def forward(self, r: Tensor, zi: Tensor, zj: Tensor):
        """
        Args:
            r: 1D tensor of distances between atoms i and j.
            zi: 1D tensor of integers. type of atom i. The choice are 0, 1, 2, ...
                the number of atom types.
            zj: 1D tensor of integers. type of atom j. The choice are 0, 1, 2, ...
                the number of atom types.

        Note:
            The shape of r, zi, and zj should be the same.

        Returns:
            A tensor of shape (len(r), n_nu). The first dimension corresponds to `nu`
            in Eq. 3 of Shapeev, and the second dimension denotes the size of the
            distances.
        """
        # shape (n_nu, len(r))
        radial = radial_basis(
            self.max_chebyshev_degree, r, r_cut=self.r_cut, envelope=self.envelope
        )

        # select c for r according to zi and zj
        c = self.c[zi, zj, :, :]  # shape(len(r), n_nu, len(degrees))

        # linear combination of radial basis functions of different degrees
        out = torch.einsum("rub, br -> ru", c, radial)

        return out


def radial_basis(
    degree: int,
    r: Tensor,
    r_min: float = 0,
    r_cut: float = 5,
    envelope: Optional[int] = None,
) -> Tensor:
    """
    Radial basis function, using Chebyshev polynomials.

    I.e. Q in Eq. 4 of Shapeev.

    Args:
        degree: max degree of the Chebyshev polynomial to use.
        r: distance, 1D tensor.
        r_min: minimum distance.
        r_cut: cutoff distance.
        envelope: envelope function to make the radial basis function smooth at r_cut.
            if None, using the MTP 2nd order polynomial envelope. Otherwise, p is a
            positive integer, and the envelope function in dimenet is used.

    Returns:
        A tensor X of shape (degree+1, *r.shape); +1 to include the zeroth degree.
        The first dimension denotes the degree of the polynomial. X[i] is the result
        for the i-th degree polynomial.
    """
    # select r < r_cut ones for computation
    mask = r < r_cut
    selected_r = r[mask]

    # normalize r to [0, 1]
    normalized_r = (selected_r - r_min) / (r_cut - r_min)

    che = chebyshev_first(degree, normalized_r)

    if envelope is None:
        env = mtp_envelope(normalized_r)
    else:
        env = dimenet_envelope(normalized_r, p=envelope)

    Q = che * env

    # prepare output
    shape = torch.Size([degree + 1]) + r.shape
    out = torch.zeros(shape, dtype=r.dtype, device=r.device)
    out[:, mask] = Q

    return out


def chebyshev_first(n: int, x: Tensor) -> Tensor:
    """Chebyshev polynomials of the first kind.

    Args:
        n: highest degree of the polynomial to compute.
        x: input tensor.

    Returns:
        A tensor of shape (n + 1, *x.shape). The first dimension denotes the degree
        of the polynomial, e.g. T[1] is the result of the first degree polynomial.
    """
    T = [torch.ones_like(x), x]  # T0 and T1
    for i in range(2, n + 1):
        T.append(2.0 * x * T[i - 1] - T[i - 2])

    T = torch.stack(T, dim=0)

    return T


def mtp_envelope(r: Tensor):
    """The envelope function used in the MTP."""
    return (1 - r) ** 2


def dimenet_envelope(r: Tensor, p: int = 6):
    """The envelope function used in DimNet.

    1 - (p+1)(p+2)/2*x**p + p*(p+2)*x**(p+1) - p*(p+1)/2*x**(p+2)

    This is also the envelope function used hybrid NN of Mingjian Wen when p = 3.
    """
    if p == 6:
        return 1 - 28 * r**6 + 48 * r**7 - 21 * r**8
    elif p == 3:
        return 1 - 10 * r**3 + 15 * r**4 - 6 * r**5
    else:
        return (
            1
            - (p + 1) * (p + 2) / 2 * r**p
            + p * (p + 2) * r ** (p + 1)
            - p * (p + 1) / 2 * r ** (p + 2)
        )
