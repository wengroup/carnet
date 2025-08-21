"""Hyper moment constructed as the tensor product of multiple atomic moments."""

import torch
from torch import Tensor, nn

from .linear import SlicedLinearCombination
from .product import TensorProduct


class HyperMoment(nn.Module):
    """
    Self tensor product of atomic moment feature tensor:

    $H = x_L \otimes x_L \otimes ... \otimes x_L$

    where $L$ is the max rank of the atomic moment feature tensor, consisting of
    natural tensors of rank 0, 1, 2, ..., and up to $L$.

    The number of $x_L$ used in the tensor product is denoted as the correlation degree
    of the hyper moment feature tensor. The hyper moment feature tensor are created
    for degree 1, 2, ..., and up max degree:
    H^1, H^2, ..., H^{max_degree}.

    For each H_i, only natural tensors of rank up to L are kept, and the ones of rank
    higher than L are discarded.

    Natural tensors of the same rank l from different H_i are linearly combined to
    generate the final hyper moment feature tensor, i.e.:
    H = \sum_d w_d H^d

    Args:
        F: Channel dimension.
        L: Max rank of the atomic moment feature tensor.
        max_out_L: Max rank for the output hyper moment feature tensor.
            If None, set to L.
        max_degree: Max correlation degree of the hyper moment feature tensor.
    """

    def __init__(
        self,
        F: int,
        L: int,
        max_out_L: int = None,
        max_degree: int = 3,
        tp_path_mode: str = "lite",
        level: int = None,
    ):
        super().__init__()

        if max_out_L is None:
            max_out_L = L
        else:
            if not max_out_L <= L:
                raise ValueError(f"Expect max_out_L <= L, got {max_out_L} > {L}.")

        self.F = F
        self.L = L
        self.max_out_L = max_out_L
        self.max_degree = max_degree
        self.tp_path_mode = tp_path_mode

        # Iterative tensor products to evaluate H^1, H^2, ..., H^{max_degree}
        # H^1 = x_L
        # H^2 = H^1 \otimes x_L
        # H^3 = H^2 \otimes x_L
        # ...
        # H^{max_degree} = H^{max_degree-1} \otimes x_L
        # So, in total there will be max_degree-1 tensor products to be evaluated.
        # For each tensor product, H^i = H^{i-1} \otimes x_L, it is chosen that the
        # maximum rank of the natural tensors in H^i is L. However, this is not the
        # case for the final output hyper moment tensor, H^{max_degree}, where the
        # maximum rank of the natural tensors is set to max_out_L.
        #
        # TODO, it might be possible that, for a given max_out_L < L, the rank of the
        #   natural tensor in H^{i} can be smaller than L. Well, maybe not. For example,
        #   if L=2, and max_out_L=0, then the scalar part of H^2 = H^1 \otimes x_2 can
        #   still come from H^1_2 and x_2. So, we need H^1 to have ranks up to L.

        self.tp = nn.ModuleList()
        for i in range(1, max_degree):
            if i == max_degree - 1:
                out_L = max_out_L
            else:
                out_L = L
            self.tp.append(
                TensorProduct(
                    F,
                    L,
                    L,
                    out_L,
                    normalize="unity",
                    path_mode=tp_path_mode,
                    level=level,
                )
            )

        # Linear combination of different degree
        # H = \sum_d w_d H^d
        if max_degree <= 1:
            # Do not need linear combination if max_degree is 1 or less
            self.register_buffer("linear_degree", None)
        else:
            self.linear_degree = SlicedLinearCombination(
                max_degree, F, [3**l for l in range(self.max_out_L + 1)]
            )

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Atomic moment tensors. Shape (..., F, T), where F is the number of
                features, and T=(3**(L+1)-1)/2 is the number of tensor components.

        Returns:
            Hyper moments. Shape (n_atoms, F, T'), where T' is the number of tensor
            components, determined by max_out_L.
        """
        assert x.shape[-1] == (3 ** (self.L + 1) - 1) // 2, "Invalid input shape."

        # The number of tensor components to keep in the output
        size = int((3 ** (self.max_out_L + 1) - 1) // 2)

        # Essentially, there is no hyper moment if max_degree is 1 or less.
        if self.max_degree <= 1:
            return x[..., :size]

        # Output hyper moments from different coupling degrees
        # Shape of each element is (..., F, T'), where T' is the size above
        out_H = [x[..., :size]]

        # TODO, given that we only need `:size` component, is it possible to enforce
        #  this in fn (namely TensorProduct)? This will make it more efficient.
        #  A: We did this for the last layer, but not for the others, yet.
        H_tmp = x
        for fn in self.tp:
            product = fn(H_tmp, x)
            H_tmp = product
            out_H.append(product[..., :size])

        out_H = torch.stack(out_H, dim=-3)  # (..., max_degree, F, T')
        H = self.linear_degree(out_H)  # (..., F, T')

        return H
