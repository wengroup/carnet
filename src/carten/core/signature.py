from collections import namedtuple
from dataclasses import dataclass

MR = namedtuple("MR", ["mul", "rank"])


class Signature(list):
    """Signature of a NaturalTensors.

    This can be though as a list of tuples, where each tuple can have a pair of numbers,
    signifying the multiplicity and rank of the tensor.

    If the multiplicity is 1, it can be omitted, and then, the tuple can be replaced
    by an integer signifying the rank of the tensor.

    Example:
        >>> Signature([(5, 0), 2, (6, 4)])
        is the same as
        >>> Signature([(5, 0), (1, 2), (6, 4)])
        meaning that there are 5 natural tensors of rank 0, 1 natural tensor of rank 2,
        and 6 natural tensors of rank 4.
    """

    def __init__(self, x: list[tuple[int, int] | int]):
        out = []
        for i, mr in enumerate(x):
            if isinstance(mr, (tuple, list)):
                if not len(mr) == 2:
                    raise ValueError(f"Input {i} `{mr}` is not a tuple of length 2.")
                out.append(MR(*mr))
            elif isinstance(mr, int):
                out.append(MR(1, mr))
            else:
                raise ValueError(f"Unexpected input {i} `{mr}`.")

        super().__init__(out)

    def simplify(self):
        """Combine consecutive MR of the same rank together."""
        out = self[0:1]
        for mr in self[1:]:
            if mr.rank == out[-1].rank:
                out[-1] = MR(out[-1].mul + mr.mul, mr.rank)
            else:
                out.append(mr)

        return Signature(out)

    def reorder(self):
        """Reorder the MR by rank."""

        # return Signature(*sorted(self, key=lambda mr: mr.rank))
        to_sort = [(i, m, r) for i, (m, r) in enumerate(self)]
        out = sorted(to_sort, key=lambda x: x[2])

        # sorted signature
        s = Signature([(m, r) for _, m, r in out])
        sig_p = [i for i, _, _ in out]

        # permutation for the corresponding data
        chunk_dims = self.chunk_dims
        data_p = [
            j
            for i, _, _ in out
            for j in range(
                sum(chunk_dims[:i]),
                sum(chunk_dims[:i]) + chunk_dims[i],
            )
        ]

        return SortedSignature(s, sig_p, data_p)

    def regroup(self):
        """Regroup the signature = reorder + simplify."""
        return self.reorder().signature.simplify()

    @property
    def dim(self) -> int:
        """The total dimension of the tensor.

        Sum of the flattened dim of all tensors.
        """
        return sum(self.chunk_dims)

    @property
    def chunk_muls(self):
        """The multiplicity of each chunk."""
        return [mr.mul for mr in self]

    @property
    def chunk_ranks(self) -> list[int]:
        """The rank of each chunk."""
        return [mr.rank for mr in self]

    @property
    def chunk_dims(self) -> list[int]:
        """The dimension of each chunk."""
        return [mr.mul * 3**mr.rank for mr in self]

    @classmethod
    def from_str(cls, s: str):
        """Construct a Signature from a string.

        The string should follow the format: `m_1 x r1 + m_2 x r_2 + ... + mn x rn`,
        where `m_i` is the multiplicity of the tensor, and `r_i` is the rank of the
        tensors.

        - A multiplicity `m_i` (and the following `x`) can be omitted if it is 1.
          For example, `1x2 + 3x4 + 5` is equivalent to `1x2 + 3x4 + 5x1`.
        - Space between `m_i`, `x`, and `r_i` is optional.

        Example:
            >>> Signature.from_str("5x0 + 2 + 6x4")
            Signature((5, 0), (1, 2), (6, 4))
        """
        s = s.split("+")

        out = []
        for mr in s:
            mr = mr.strip()
            if "x" in mr:
                m, r = mr.split("x")
                out.append(MR(int(m.strip()), int(r.strip())))
            else:
                out.append(MR(1, int(mr)))

        return cls(out)

    @classmethod
    def from_ranks(cls, ranks: list[int]):
        """
        The number of consecutive tensors of the same rank will be multiplicity
        of that rank.

        Example:
            >>>Signature.from_ranks([0,1,1,1,2,2,1,1,1,1])
            Signature((1, 0), (3, 1), (2, 2), (4, 1))
        """
        return cls(ranks).simplify()

    def __str__(self):
        return f"{self.__class__.__name__}({super().__repr__()})"

    def __contains__(self, r: int):
        """Check if a rank is in the signature."""
        return r in self.chunk_ranks


@dataclass
class SortedSignature:
    """
    A sorted signature.

    Args:
        signature: Sorted signature.
        sig_perm: The permutation to sort the signature.
        data_perm: The permutation to sort the data.
    """

    signature: Signature
    sig_perm: list[int]
    data_perm: list[int]
