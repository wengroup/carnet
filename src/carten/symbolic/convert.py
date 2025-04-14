"""Convertion between ordinary cartesian tensor T and natural tensors X."""

import torch
import torch.nn as nn
from torch import Tensor

from carten.symbolic.tabulate import get_G_H_S


class Converter(nn.Module):
    """
    Convertor to map between ordinary cartesian tensor T and natural tensors X.

    The conversion is done as follows:
    X = H T
    T'= G T

    Args:
        rank: The rank of the tensor T.
        symmetry: symmetry of the Cartesian ordinary tensor, if any. For example,
            - "ij=ji" means that the tensor is a fully symmetric rank-2 tensor (e.g.
                stress tensor);
            - "ijk=ikj" means that the tensor is a rank-3 tensor with the last two
                indices symmetric (e.g. piezoelectric tensor);
            - "ijk=ikj=jik" means that the tensor is a fully symmetric rank-3 tensor;
            - "ijkl=jikl=klij" means that the tensor is a rank-4 tensor with both minor
                symmetry (between i and j, and between k and l) and major symmetry (
                between ij and kl). For example, the elastic tensor has this symmetry.
            The number of unique letters gives the rank of the tensor (what letters to
            use does not matter).
    """

    def __init__(self, rank: int, symmetry: str = None):
        super().__init__()
        self.rank = rank
        self.symmetry = symmetry

        out = get_G_H_S(rank, symmetry)

        self.j_num_p = {}
        for j, out_j in out.items():
            self.j_num_p[j] = len(out_j["G"])
            # TODO, the tensors of different G_j_p might be batched, which can
            #  accelerate the computation
            for p, (G, H) in enumerate(zip(out_j["G"], out_j["H"])):
                self.register_buffer(f"G_{j}_{p}", torch.tensor(G["numerical"]))
                self.register_buffer(f"H_{j}_{p}", torch.tensor(H["numerical"]))
                setattr(self, f"G_{j}_{p}_rule", G["rule"])
                setattr(self, f"H_{j}_{p}_rule", H["rule"])

    # TODO, the looping is very inefficient, need to think about better ways
    def to_natural_tensor(self, T: Tensor) -> dict[int, Tensor]:
        """
        Convert an ordinary cartesian tensor T to natural tensors X.

        Args:
            T: The tensor to convert.

        Returns:
            A dictionary {j: X} where j is the rank (j=0, 1, ...n) and X is the
            corresponding natural tensor. The shape of X is (N, 3,...,3) where N is
            the seniority of rank-j natural tensor, and there are j (rank) of 3s.
            In other words, the first dim batches natural tensor of the same rank j,
            but different seniority p.
        """
        out = {}
        for j, num_p in self.j_num_p.items():
            out[j] = [
                torch.einsum(
                    getattr(self, f"H_{j}_{p}_rule"), getattr(self, f"H_{j}_{p}"), T
                )
                for p in range(num_p)
            ]

        return out

    # TODO, the looping is very inefficient, need to think about better ways
    def to_ordinary_tensor(self, X: dict[int, Tensor]) -> Tensor:
        """
        Convert natural tensors X to ordinary cartesian tensor T.

        This is the inverse of `to_natural_tensor`.

        Args:
            X: A dictionary {j: X} where j is the rank (j=0, 1, ...n) and X is the
            corresponding natural tensor. The shape of X is (N, 3,...,3) where N is
            the seniority of rank-j natural tensor, and there are j (rank) of 3s.
            In other words, the first dim batches natural tensor of the same rank j,
            but different seniority p.

        Returns:
            An ordinary cartesian tensor T of rank n.
        """
        out = []
        for j, num_p in self.j_num_p.items():
            out.extend(
                [
                    torch.einsum(
                        getattr(self, f"G_{j}_{p}_rule"),
                        getattr(self, f"G_{j}_{p}"),
                        X[j][p],
                    )
                    for p in range(num_p)
                ]
            )

        out = torch.sum(torch.stack(out), dim=0)

        return out
