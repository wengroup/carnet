"""Radial basis functions."""

from typing import Optional

import torch
from torch import Tensor


@torch.jit.script
def chebyshev_first(n: int, x: Tensor) -> Tensor:
    """Chebyshev polynomials of the first kind.

    Args:
        n: highest degree of the polynomial to compute.
        x: input tensor.

    Returns:
        A tensor of shape (*x.shape, n+1). The last dimension denotes the degree
        of the polynomial, e.g. T[:,1] is the result of the first degree polynomial.
    """
    T = [torch.ones_like(x), x]  # T0 and T1
    for i in range(2, n + 1):
        T.append(2.0 * x * T[i - 1] - T[i - 2])

    T = torch.stack(T, dim=-1)

    return T


def mtp_envelope(r: Tensor):
    """The envelope function used in the MTP."""
    return (1 - r) ** 2


@torch.jit.script
def dimenet_envelope(r: Tensor, p: int = 6):
    """The envelope function used in DimNet.

    1 - (p+1)(p+2)/2*x**p + p*(p+2)*x**(p+1) - p*(p+1)/2*x**(p+2)

    Args:
        r: normalized distance, in the range [0, 1].

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


@torch.jit.script
def radial_basis(
    r: Tensor,
    degree: int,
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
        A tensor X of shape (*r.shape, degree+1); +1 to include the zeroth degree.
        The last dimension denotes the degree of the polynomial. X[..., i] is the result
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

    Q = che * env.unsqueeze(-1)

    # prepare output
    shape = r.shape + (degree + 1,)
    out = torch.zeros(shape, dtype=r.dtype, device=r.device)
    out[mask, :] = Q

    return out
