"""Hyper moment constructed as the tensor product of multiple atomic moments."""

import torch
from torch import Tensor, nn

from .linear import SlicedLinearCombination
from .product import TensorProduct


class HyperMoment(nn.Module):
    r"""
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
        tp_path_polar_only: bool = False,
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
                    path_polar_only=tp_path_polar_only,
                    level=level,
                )
            )

        # Linear combination of different degree
        # H = \sum_d w_d H^d
        if max_degree <= 1:
            self.register_buffer("linear_degree", None)
        else:
            self.linear_degree = SlicedLinearCombination(
                max_degree, F, [3**l for l in range(self.max_out_L + 1)]
            )

        # Precompute the number of tensor components to keep in the output
        self.size = int((3 ** (self.max_out_L + 1) - 1) // 2)

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

        # Essentially, there is no hyper moment if max_degree is 1 or less.
        if self.max_degree <= 1:
            return x[..., : self.size]

        # # Shape of each element is (..., F, T'), where T' is the size above
        # out_H = [x[..., :self.size]]
        #
        # H_tmp = x
        # for fn in self.tp:
        #     product = fn(H_tmp, x)
        #     H_tmp = product
        #     out_H.append(product[..., :self.size])
        #
        # out_H = torch.stack(out_H, dim=-3)  # (..., max_degree, F, T')
        # H = self.linear_degree(out_H)  # (..., F, T')
        #
        # return H

        # Below is an equivalent but more memory efficient implementation. out_H takes
        # up a lot of memory when max_degree is large.

        # Degree d=1 contribution
        H_final = self._get_degree_weights(1) * x[..., : self.size]

        H_tmp = x
        for d in range(2, self.max_degree + 1):
            H_tmp = self.tp[d - 2](H_tmp, x)
            current_H = H_tmp[..., : self.size]

            # Weights for degree d: (F, T')
            w_d = self._get_degree_weights(d)

            # (F, T') * (..., F, T') -> (..., F, T')
            H_final += w_d * current_H

        return H_final

    def _get_degree_weights(self, d: int) -> Tensor:
        """
        Extract weights for a specific correlation degree d.

        Args:
            d: Correlation degree (1-based index).

        Returns:
            Weights of shape (F, T').
        """
        # linear_degree.weight is (num_slices, max_degree, F)
        # weight_map maps T' components to slice indices
        # Result w is (T', F), transposed to (F, T')
        w = self.linear_degree.weight[self.linear_degree.weight_map, d - 1, :]
        return w.t()

