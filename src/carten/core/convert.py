"""Convertion between ordinary cartesian tensor T and natural tensors X."""

import torch
import torch.nn as nn
from natt.GHS import get_G_H_S
from torch import Tensor


class Converter(nn.Module):
    """
    Convertor to map between ordinary cartesian tensor T and natural tensors X.

    The conversion is done as follows:
    X = H T
    T'= G T

    Args:
        symmetry: symmetry of the Cartesian ordinary tensor. For example,
            - "ij" means a rank-2 tensor without any symmetry;
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

    def __init__(self, symmetry: str):
        super().__init__()
        self.rank = len(set(symmetry.replace("=", "").replace(" ", "")))

        if "=" not in symmetry:
            self.symmetry = None
        else:
            self.symmetry = symmetry

        out = get_G_H_S(self.rank, symmetry)

        self.l_num_p = {}
        for l, out_l in out.items():
            self.l_num_p[l] = len(out_l["G"])
            # TODO, the tensors of different G_j_p might be batched, which can
            #  accelerate the computation
            for p, (G, H) in enumerate(zip(out_l["G"], out_l["H"])):
                self.register_buffer(f"G_{l}_{p}", torch.tensor(G["numerical"]))
                self.register_buffer(f"H_{l}_{p}", torch.tensor(H["numerical"]))
                setattr(self, f"G_{l}_{p}_rule", G["rule"])
                setattr(self, f"H_{l}_{p}_rule", H["rule"])

    def to_natural_tensor(self, T: Tensor) -> dict[int, Tensor]:
        """
        Convert an ordinary cartesian tensor T to natural tensors X.

        Args:
            T: The tensor to convert. Shape (B, 3, ..., 3), where B represents arbitrary
                batch dimensions, and the number of 3s is, of course, equal to the rank
                of the tensor.

        Returns:
            Natural tensors {l: X}, where l is the rank (l=0, 1, ...n) and X is the
            corresponding tensor value. The shape of X is (B, F, 3^l) where B represents
            arbitrary batch dimensions, F is the number of natural tensors of rank l,
            and 3^l is the flattened dimension of the tensor. The second dimension F
            batches natural tensors of the same rank l, but different seniority p.
        """
        B = T.shape[: -self.rank]

        # TODO, the looping is very inefficient, need to think about better ways
        out = {}
        for l, num_p in self.l_num_p.items():
            out[l] = torch.stack(
                [
                    torch.einsum(
                        getattr(self, f"H_{l}_{p}_rule"), getattr(self, f"H_{l}_{p}"), T
                    ).reshape(*B, 3**l)
                    for p in range(num_p)
                ],
                dim=-2,  # stack to create the new F dimension
            )

        return out

    def to_ordinary_tensor(
        self, X: dict[int, Tensor], flatten_tensor_dim: bool = False
    ) -> Tensor:
        """
        Convert natural tensors X to ordinary cartesian tensor T.

        This is the inverse of `to_natural_tensor`.

        Args:
            X: Natural tensors {l: X}, where l is the rank (l=0, 1, ...n) and X is the
                corresponding tensor value. The shape of X is (B, F, 3^l) where B
                represents arbitrary batch dimensions, F is the number of natural
                tensors of rank l, and 3^l is the flattened dimension of the tensor.
                The second dimension F batches natural tensors of the same rank l,
                but different seniority p.
            flatten_tensor_dim: If True, the output tensor will have the last dimension
                `rank` tensor flattened to a single dimension of size `3**rank`.

        Returns:
            An ordinary cartesian tensor T corresponding to the natural tensors X.
            The shape is (B, 3, ..., 3), where B represents arbitrary batch dimensions,
            and the number of 3s is, of course, equal to the rank of the tensor.
            If `flatten_tensor_dim` is True, the tensor dim will be flattened to a
            single dimension of size `3**rank`.
        """
        B = X[list(X.keys())[0]].shape[:-2]

        out = []
        # TODO, the looping is very inefficient, need to think about better ways
        for l, num_p in self.l_num_p.items():
            for p in range(num_p):
                rule = getattr(self, f"G_{l}_{p}_rule")
                G = getattr(self, f"G_{l}_{p}")
                X_ = X[l][..., p, :]

                # special rule for scalars
                if l == 0:
                    X_ = X_[..., 0]
                else:
                    X_ = X_.reshape(*B, *(3,) * l)

                prod = torch.einsum(rule, G, X_)

                if flatten_tensor_dim:
                    prod = prod.reshape(*B, 3**self.rank)

                out.append(prod)

        out = torch.sum(torch.stack(out), dim=0)

        return out
