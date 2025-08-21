"""
Product of natural tensors.
"""

from typing import Optional

import torch
from torch import Tensor, nn

from carten.core.tp import tp_even, tp_odd
from carten.core.tp_with_weight import tp_even_with_weight, tp_odd_with_weight
from carten.module.linear import LinearCombination
from carten.module.utils import check_rank, get_paths


class TensorProduct(nn.Module):
    """
    Tensor product of two feature tensors.

    z_L3 = x_L1 \otimes y_L2

    Multiple paths l1, l2 -> l3 can lead to the same l3, and these paths are linearly
    combined to get the output tensor of rank l3.


    A feature tensor is a collection of natural tensors with rank 0, 1, and up to L.
    It has the shape (..., F, T), where ... can be any batching dimensions, `F` is the
    number of features (channels) and `T = \sum_{l=0}^L 3**l= (3**(L+1)-1)/2` is the
    tensor dimension. The tensor dimension combines all elements of the natural tensors
    up to rank L.

    In other words, the innermost dim `T` of a feature tensor looks like:
    X_0, X_1^0, X_1^1, X_1^2, X_2^00, X_2^01, X_2^02, X_2^10, X_2^11, X_2^12, X_2^20, X_2^21, X_2^22, ...
    where subscripts denote the rank of the natural tensor and superscripts denote the
    index of the element.

    The innermost dim `T` can be sliced to get the natural tensors of rank l:
    x_l = x[..., [3**l-1]//2 : [3**(l+1)-1]//2]
    """

    def __init__(
        self,
        F: int,
        L1: int,
        L2: int,
        L3: int | list[int] = None,
        normalize: str = "unity",
        path_mode: str = "full",
        level: int = None,
        for_atomic_moment: bool = False,
    ):
        """
        Args:
            F: number of features (channel dimension) in the input feature tensors.
            L1: maximum rank of the natural tensor in the first input feature tensor.
            L2: maximum rank of the natural tensor in the second input feature tensor.
            L3: rank of the output feature tensor.
                If int, it is the maximum allowed rank of the natural tensor in the
                output feature tensor. Then the output feature tensor will consist of
                natural tensors of rank 0, 1, ..., L3.
                If a tuple or list, only natural tensors of ranks in the tuple will be
                included in the output feature tensor.
                If None, the output feature tensor will consist of natural tensors of
                rank up to the maximum rank of the input feature tensors, namely
                0, 1, ..., max(L1, L2).
            normalize: normalization method for the output tensor. Options are:
                `unity` or `none`. See `carten.core.tp.tp_even`.
            path_mode: mode to construct the paths from L1 and L2 to L3. Options are:
                `full`, `camp`, or `lite`. See `carten.module.utils.get_paths`.
            level: If path_mode is `level`, this is the maximum level of the paths.
            for_atomic_moment: If False, this is for a general tensor product,
                which is simply a wrapper around the tp_even and tp_odd functions.
                When it is True, it is used in the context of atomic moments,
                Z= RXY, where X is of shape (..., F, T1) and Y is of shape (..., T2),
                and R is a dictionary of additional weights, of shape (..., F).
        """
        super().__init__()
        self.L1 = L1
        self.L2 = L2
        self.L3 = check_rank(L1, L2, L3)
        self.normalize = normalize
        self.path_mode = path_mode
        self.for_atomic_moment = for_atomic_moment

        # Set default value for level
        if path_mode == "level" and level is None:
            level = max(L1, L2)

        self.paths = get_paths(self.L1, self.L2, self.L3, path_mode, level)

        # Kernel parameters for linear combination of paths to each l3
        # Each (l1, l2, l3) has its own kernel parameters
        self.kernels = nn.ModuleList()
        for l3 in self.L3:
            n = len(self.paths[l3])
            if n == 1:
                # If only one path, no need to do a linear combination.
                # None as a placeholder, but never used below in forward()
                self.kernels.append(None)
            else:
                self.kernels.append(LinearCombination(n, F))

        self.z_tensor_dims = [3**l3 for l3 in self.L3]

    def forward(
        self, x: Tensor, y: Tensor, R: Optional[dict[str, Tensor]] = None
    ) -> Tensor:
        """
        Evaluate the tensor product of two feature tensors:
        z_l3 = R_l1l2l3 x_l1 \otimes y_l2

        Args:
            x: Feature tensor of maximum rank L1. Shape (..., F, T1),
                where T1 = \sum_{l1} 3**l1.
            y: Feature tensor of maximum rank L2. Shape (..., F, T2), or Shape(..., T2),
                where T2 = \sum_{l2} 3**l2. The shape depends on `for_atomic_moment`.
            R: additional parameters to be multiplied with the tensor product. If None,
                the tensor product is evaluated without additional parameters.
                Shape (..., F), where F is the number of features.

        Returns:
            z: Output feature tensor, whose ranks are determined the input L3. Shape
                (..., F, T3), where T3 = \sum_{l3} 3**l3.
        """
        if self.for_atomic_moment:
            assert R is not None, "Weights are needed for atomic moment, but R is None"

        z = []
        for idx, kernel in enumerate(self.kernels):
            l3 = self.L3[idx]
            z_l3 = []  # z_l3 from all paths
            for path in self.paths[l3]:
                l1, l2, _ = path

                # x_l1: (..., F, 3**l1)
                # y_l2: (..., F, 3**l2) or (..., 3**l2)
                # int() needed for TorchScript
                x_l1 = x[..., int((3**l1 - 1) // 2) : int((3 ** (l1 + 1) - 1) // 2)]
                y_l2 = y[..., int((3**l2 - 1) // 2) : int((3 ** (l2 + 1) - 1) // 2)]

                if self.for_atomic_moment:
                    w = R[str(path)]  # (..., F)

                    if (l1 + l2 - l3) % 2 == 0:
                        z_tmp = tp_even_with_weight(
                            x_l1, y_l2, w, l1, l2, l3, self.normalize
                        )
                    else:
                        z_tmp = tp_odd_with_weight(
                            x_l1, y_l2, w, l1, l2, l3, self.normalize
                        )

                else:
                    # z_tmp: (..., F, 3**l3)
                    if (l1 + l2 - l3) % 2 == 0:
                        z_tmp = tp_even(x_l1, y_l2, l1, l2, l3, self.normalize)
                    else:
                        z_tmp = tp_odd(x_l1, y_l2, l1, l2, l3, self.normalize)

                z_l3.append(z_tmp)  # list of tensors of shape (..., F, 3**l3)

            # Only one path to l3
            if len(z_l3) == 1:
                z_l3_combined = z_l3[0]  # (..., F, 3**l3)
            # Multiple paths to l3
            else:
                z_l3 = torch.stack(z_l3, dim=-3)  # (..., Np, F, 3**l3)
                # (..., F, 3**l3) Linear combination of all Np paths to l3
                z_l3_combined = kernel(z_l3)

            z.append(z_l3_combined)

        # Combine z of different l3 in the last dim
        z = torch.cat(z, dim=-1)  # (..., F, T3)

        return z
