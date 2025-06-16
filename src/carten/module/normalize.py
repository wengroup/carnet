""" Normalization function and layer. """

import torch
from torch import Tensor, nn


class LayerNorm(nn.Module):

    def __init__(
        self,
        dim: int,
        slice_sizes: list[int],
        affine: bool = True,
        normalization: str = "component",
        eps: float = 1e-5,
    ):
        """
        Layer normalization layer.

        For a feature tensor of shape (..., F, T), this layer normalizes the features
        using statistics computed across the channel dimension (F).
        Statistics are computed separately for each slice across the T dimension.

        For scalars, the normalization is
        x = [(x - mean) / sqrt(var + eps)] * a + b , where a and b are learnable
        parameters.

        For vectors, the normalization is
        x = [x / sqrt(var + eps)] * a, where a is learnable parameters.

        This is the same layer norm used in equiformer.


        Args:
            dim: feature dimension (F).
            slice_sizes: Sizes of slices across the T dimension, 3**0, 3**1, ...,
                3**(n-1) for n slices.
            affine: whether to use affine transformations, i.e. a and b values.
            normalization: The method to compute `var`. If`component`, it is normalized
                by individual components; if `norm`, it is normalized by the sum of the
                features across the channel dimension.
            eps: A small value added to the denominator for numerical stability.
        """
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.slice_sizes = torch.tensor(slice_sizes)
        self.affine = affine
        self.normalization = normalization

        if affine:
            self.a = nn.Parameter(torch.ones(dim, len(slice_sizes)))
            if self.slice_sizes[0] != 1:
                raise ValueError(
                    f"Affine transformation can only be used when the first slice "
                    f"size is 1 (namely a scalar), but got {self.slice_sizes[0]}"
                )
            self.b = nn.Parameter(torch.zeros(dim, 1))
        else:
            self.register_parameter("a", None)
            self.register_parameter("b", None)

    def forward(self, input: Tensor):

        # shape of input: (..., F, T)

        # For each feature slice:
        # 1. compute norm (when `norm`) or norm dividing by number of elements of each
        # slice across the T' dimension;
        # 2. compute the mean of the norms across the feature dim F.
        # 3. compute the scaling factor as the square root of the mean norm plus eps;
        # 4. normalize the slice by the scaling factor.
        out = []
        start = 0
        for size in self.slice_sizes:
            end = start + size
            data = input[..., :, start:end]  # (..., F, size)

            if size == 1:
                # Compute mean across feature dimension (F) for scalar. This assumes input 0 is
                # a scalar, which should be the case.
                mean = data.mean(dim=-2, keepdim=True)  # (..., 1, 1)
                data = data - mean

            if self.normalization == "component":
                # (..., F, 1)
                norm = data.pow(2).mean(dim=-1, keepdim=True)
            elif self.normalization == "norm":
                # (..., F, 1)
                norm = data.pow(2).sum(dim=-1, keepdim=True)
            else:
                raise ValueError()

            # (..., 1, 1)
            scaling = torch.sqrt(norm.mean(dim=-2, keepdim=True) + self.eps)

            data = data / scaling  # (..., F, slice_size)
            out.append(data)

            start = end

        out = torch.cat(out, dim=-1)  # (..., F, T)

        if self.affine:
            # `a` is of the shape (F, len(slice_sizes)), we make it of the shape
            # (F, T), where T is the total number of tensor components, by repeating
            # it across the T dimension according to the slice sizes.
            a = torch.repeat_interleave(self.a, self.slice_sizes, dim=1)
            out *= a
            out[..., :, 0:1] += self.b

        return out
