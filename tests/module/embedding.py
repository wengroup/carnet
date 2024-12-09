import torch
from torch import Tensor, nn


class Embedding(nn.Module):
    """
    Embed integers 0, 1, 2, ... as learnable fixed-size vectors.

    Args:
        size: number of integers to embed.
        embedding_dim: output dim of the species embedding.
    """

    def __init__(self, size: int, embedding_dim: int = 8):
        super().__init__()

        self.size = size
        self.embedding_dim = embedding_dim

        self.linear = nn.Linear(size, embedding_dim)

        self.dtype = torch.get_default_dtype()

    def forward(self, input: Tensor) -> Tensor:
        """
        Args:
            input: Input tensor of integers, each integer should be in [0, size).
                Shape (N,).

        Returns:
            Embedded integers. Shape (N, embedding_dim).
        """
        one_hot = torch.nn.functional.one_hot(input, self.size).to(self.dtype)
        embedding = self.linear(one_hot)

        return embedding
