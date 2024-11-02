"""
The aims to find the relationships between tensors with contracted with delta or epsilon
tensors.

This based on:
A NOTE ON THE DECOMPOSITION OF TENSORS INTO TRACELESS SYMMETRIC TENSORS,
A. J. M. Spencer, 1970

The relationship is given in eq 13.
"""
from carten.utils import letter_index
import itertools


def get_a(n: int, r: int) -> list[str]:
    """
    Permute the indices of a tensor.

    a_(p,a,b,r-2)q,r+1...n in eq. 10.

    We use a for i1, b for i2, ...

    Returns:
        All permutations of the indices: a_(p,a,b,r-2)q,r+1...n
    """
    indices = "p" + letter_index(r - 2)

    a_perm = ["".join(p) for p in itertools.permutations(indices)]

    appendix = letter_index(n - r, start=r + 1)

    out = [p + "q" + appendix for p in a_perm]

    return out


def get_b(a: list[str]) -> list[str]:
    """
    Get b in Eq 10.

    b = epsilon_pqk a_(p,a,b,r-2)q,r+1...n

    Args:
        a:

    Returns:

    """


if __name__ == "__main__":
    a = get_a(6, 4)

    tmp = 1
