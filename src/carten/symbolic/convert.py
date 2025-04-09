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
    """

    def __init__(self, rank: int):
        super().__init__()

        self.rank = rank
        out = get_G_H_S(rank)

        self.j_num_p = {}
        for j, out_j in out.items():
            self.j_num_p[j] = len(out_j["G"])
            for p, (G, H) in enumerate(zip(out_j["G"], out_j["H"])):
                self.register_buffer(f"G_{j}_{p}", torch.tensor(G["numerical"]))
                self.register_buffer(f"H_{j}_{p}", torch.tensor(H["numerical"]))
                setattr(self, f"G_{j}_{p}_rule", G["rule"])
                setattr(self, f"H_{j}_{p}_rule", H["rule"])

    # TODO, the looping is very inefficient, need to think about better ways
    def to_natural_tensor(self, T: Tensor) -> dict[int, list[Tensor]]:
        """
        Convert an ordinary cartesian tensor T to natural tensors X.

        Args:
            T: The tensor to convert.

        Returns:
            A dictionary {j: [X]} where j is the rank of the natural tensor, j=0,1,
            ...n and X are the natural tensors. For each j, there can be multiple
            X tensors.
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
    def to_ordinary_tensor(self, X: dict[int, list[Tensor]]) -> Tensor:
        """
        Convert natural tensors X to ordinary cartesian tensor T.

        This is the inverse of `to_natural_tensor`.

        Args:
            A dictionary {j: [X]} where j is the rank of the natural tensor, j=0,1,
            ...n and X are the natural tensors. For each j, there can be multiple
            X tensors.

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
