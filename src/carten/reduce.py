"""Helper functions to decompose the special tensors or tensor product of two tensors
into natural tensors."""
import itertools
import warnings

import torch
from torch import Tensor

from carten.natural_tensor import NaturalTensors
from carten.utils import dij, double_index, eijk, letter_index, repeat_double_index


def reduce_tensor(
    t: Tensor, symmetry: str = None, start_dim: int = 0
) -> NaturalTensors:
    """
    Decompose a general tensor into natural tensors.

    The reduction spectrum can be determined recursively by the symmetry of the indices.

    Args:
        t: the input tensor
        symmetry: a string that describes the symmetry of the indices. For example,
            `aabb` means the first two indices are symmetric, and next two indices are
            symmetric. Default is None, which means there is no symmetry.
        start_dim: the starting dimension to perform the operation. Dimensions before
            `start_dim` will not be used in the operation.

    Returns:
        A NaturalTensors.
    """


def reduce_symmetric_tensor(u: Tensor, start_dim: int = 0) -> NaturalTensors:
    """
    Decompose a fully symmetric tensor into natural tensors.

    Args:
        u: a symmetric tensor
        start_dim: the starting dimension to perform the operation. Dimensions before
            `start_dim` will not be used in the operation.

    Reference:
        1. http://dx.doi.org/10.1080/00018737800101454

    Returns:
        A NaturalTensors. Let m = u.ndim - start_dim, if n is even, there would be
        m/2 + 1 natural tensors of ranks m, m-2, ..., 2, 0; if n is odd, there would be
        (m+1)/2 natural tensors of rank m, m-2, ..., 3, 1.
    """

    def get_rule(indices: str, num_delta: int):
        delta_indices = ",".join([indices[2 * i : 2 * i + 2] for i in range(num_delta)])
        right = indices[num_delta * 2 :]
        return f"...{indices},{delta_indices}->...{right}"

    m = u.ndim - start_dim  # rank of the tensor
    D = m // 2  # maximum number of deltas that can be contracted

    indices = letter_index(m)
    d = dij(u.device)

    output = [remove_trace(u, start_dim=start_dim)]
    for i in range(1, D + 1):
        rule = get_rule(indices, i)
        v = torch.einsum(rule, u, *([d] * i))
        traceless = remove_trace(v, start_dim=start_dim)
        output.append(traceless)

    return NaturalTensors.from_sequence(output, start_dim=start_dim)


def get_dyadic_tensor(r: Tensor, rank: int = 2, normalize: bool = True) -> Tensor:
    r"""
    Create a generalized dyadic tensor.

    For rank = 0, the dyadic tensor is a scalar, simply equal to 1.
    For rank = 1, the dyadic tensor is a vector, simply equal to r.
    For rank >= 2, the generalized dyadic tensor is the tensor product of the vector r
    with itself, i.e. :math:`r \otimes r \otimes \cdots \otimes r`. The rank is the
    number of vectors in the tensor product.

    Args:
        r: shape (..., 3) the vector to construct the generalized dyadic tensor. Only
            the last dimension is used to construct the tensor. The ellipsis represents
            any number of dimensions that allows batching.
        rank: rank of the generalized dyadic tensor, i.e. the number of times to tensor
            product the vector r with itself. Rank must be greater than or equal to 1.
        normalize: whether to normalize the vector r as a unit vector before
            constructing the generalized dyadic tensor.

    Returns:
        A tensor of shape (..., 3, 3, ..., 3), where the ... represents the batching
        dimensions, and the number of 3's is equal to the rank.
    """
    if rank < 0:
        raise ValueError("Rank must be greater than or equal to 0.")
    elif rank == 0:
        shape = r.shape[:-1]
        return torch.ones(*shape).to(r.device)
    else:
        if normalize:
            norm = torch.norm(r, dim=-1, keepdim=True)
            if torch.any(norm < 1e-3):
                warnings.warn("The norm of the vector(s) is smaller than 1e-3.")
            r = r / norm

        indices = letter_index(rank)
        data = [r] * rank
        t = torch.einsum(f"{','.join(['...'+i for i in indices])}->...{indices}", *data)

        return t


def reduce_dyadic_tensor(
    r: Tensor, rank: int = 2, normalize: bool = True
) -> NaturalTensors:
    r"""
    Decompose a generalized dyadic tensor into natural tensors.

    Args:
        r: shape (..., 3) the vector to construct the generalized dyadic tensor. The
            ellipsis represents any number of dimensions that allows batching.
        rank: rank of the generalized dyadic tensor, i.e. the number of times to tensor
            product the vector r with itself.
        normalize: whether to normalize the vector r as a unit vector before
            constructing the generalized dyadic tensor.
    """
    U = get_dyadic_tensor(r, rank=rank, normalize=normalize)

    return reduce_symmetric_tensor(U)


def symmetrize_and_remove_trace(
    t: Tensor, start_dim: int = 0, symmetry: str = None
) -> Tensor:
    """
    Symmetrize and remove the trace of a (generic) tensor.

    This only extracts the symmetric traceless part of the tensor at the same rank.
    The antisymmetric part is totally ignored, which can further be decomposed into
    symmetric and traceless tensors of lower ranks.

    Args:
        t: input tensor
        start_dim: the starting dimension to perform the operation. Dimensions before
            `start_dim` will not be used in the operation.
        symmetry: a string that describes the symmetry of the indices. For example,
            `abba` means the first and the fourth indices are symmetric, and the
            second and the third indices are symmetric. Default is None, which means
            there is no symmetry between

    Returns:
        A symmetric traceless tensor of the same rank as the input tensor.
    """
    return remove_trace(symmetrize(t, start_dim, symmetry), start_dim)


def symmetrize(t: Tensor, start_dim: int = 0, symmetry: str = None) -> Tensor:
    """
    Symmetrize a tensor.

    The symmetrization is done by averaging over unique permutations of the indices,
    considering the symmetry of the indices.

    Args:
        t: The tensor to symmetrize
        start_dim: the starting dimension to perform the operation. Dimensions before
            `start_dim` will not be used in the operation.
        symmetry: A string that describes the symmetry of the indices. For example,
            `abba` means the first and the fourth indices are symmetric, and the
            second and the third indices are symmetric. Default is None, which means
            there is no symmetry between the indices.

    Returns:
        The symmetrized tensor.
    """

    # fully symmetrize the tensor
    if symmetry is None:
        permutations = itertools.permutations(range(start_dim, t.ndim))
        if start_dim > 0:
            prefix = list(range(start_dim))
            permutations = [prefix + list(p) for p in permutations]

    # symmetrize with the given symmetry
    else:
        assert (
            start_dim + len(symmetry) == t.ndim
        ), "The length of the symmetry string must match the tensor shape."
        permutations = get_permutations(symmetry, start_dim)

    u = torch.mean(torch.stack([t.permute(p) for p in permutations]), dim=0)

    return u


def symmetrize_2(t: Tensor, num_delta: int, start_dim: int = 0) -> Tensor:
    """
    Symmetrize a tensor that is obtained by contracting a symmetric tensor with deltas.

    Symmetrization is done by summation over unique permutations of the indices,
    considering the three set of symmetries. See `get_permutations_2` for more details.

    Args:
        t: the tensor
        num_delta: number of deltas used to obtain the tensor
        start_dim: the starting dimension to perform the operation. Dimensions before
            `start_dim` will not be used in the operation.

    Returns:
        A symmetrized tensor.
    """
    # rank of the symmetric tensor, after considering the start_dim
    m = t.ndim - start_dim

    # Get unique permutations
    permutations = get_permutations_2(m, num_delta, start_dim)

    # Sum over the permutations
    u = torch.sum(torch.stack([t.permute(p) for p in permutations]), dim=0)

    return u


def remove_trace(u: Tensor, start_dim: int = 0) -> Tensor:
    """
    Remove the trace of a symmetric tensors to get a natural tensor of the same rank.

    Args:
        u: a fully symmetric tensor
        start_dim: the starting dimension to perform the operation. Dimensions before
            `start_dim` will not be used in the operation.

    This implements:
    X_{ij\dots m} = U_{ij\dots m} + \sum_{d=1}^D (-1)^d \frac{(2m-2d-1)!!}{(2m-1)!!}
    \{ \delta_{ij}\delta_{kl} \dots U_{rrss\dots m} \}
    where:
    U: fully symmetric tensor of rank m
    X: natural tensor of rank m
    D: D=m/2 if m is even, D=(m-1)/2 if m is odd
    {} denotes fully symmetrization.

    References:
        1. Eq 10 of http://dx.doi.org/10.1080/00018737800101454
        2. Cartesian tensors writeup by Mingjian Wen, which is an explicit form of the
           above reference.

    Returns:
        A natural tensor of the same shape as the input tensor.
    """
    device = u.device

    m = u.ndim - start_dim
    D = m // 2

    delta = dij(device)
    coeff = 1
    out = u
    for d in range(1, D + 1):
        rule = remove_trace_rule(m, d)

        # Contract with multiple deltas to get a tensor of the same rank as u
        prod = torch.einsum(rule, u, *([delta] * d))

        prod = symmetrize_2(prod, num_delta=d, start_dim=start_dim)

        # coeff = (-1) ** d / double_factorial(2 * m - 1, 2 * m - 2 * d - 1 + 2, device)
        coeff = -coeff / (2 * m - 2 * d + 1)

        out = out + coeff * prod

    return out


def remove_trace_rule(m: int, d: int) -> str:
    """
    Get the contraction rule to remove the trace of a symmetric tensor.

    Note, d <= m/2.

    Args:
        m: rank of the symmetric tensor
        d: the number of delta

    Returns:
        rule: the contraction rule
        symmetry: the symmetry of the indices
    """
    u_contracted = "".join(repeat_double_index(d))
    u_remain = letter_index(m - 2 * d, start=d)
    delta = double_index(d, start=m - d)

    return (
        f"...{u_contracted}{u_remain},{','.join(delta)}->...{u_remain}{''.join(delta)}"
    )


def get_permutations(symmetry: str, start_dim: int = 0) -> list[list[int]]:
    """
    Get the unique permutations of the indices for symmetrizing a tensor.

    This works for the case where part or all of the indices are symmetric.

    Args:
        symmetry: A string that describes the symmetry of the indices. For example,
            `abba` means the first and the fourth indices are symmetric, and the
            second and the third indices are symmetric. The symmetry only applies to
            the indices after the `start_dim`.
        start_dim: the starting dimension to perform the operation. Dimensions before
            `start_dim` will not be used in the operation.

    Example:
        >>> get_permutations('abba')
        [[0, 1, 2, 3],  # abba
         [0, 1, 3, 2],  # abab
         [0, 3, 1, 2],  # aabb
         [1, 0, 2, 3],  # baba
         [1, 0, 3, 2],  # baab
         [1, 2, 0, 3]]  # bbaa
        >>> get_permutations('abba', 2)
        [[0, 1, 2, 3, 4, 5],  # abba
         [0, 1, 2, 3, 5, 4],  # abab
         [0, 1, 2, 5, 3, 4],  # aabb
         [0, 1, 3, 2, 4, 5],  # baba
         [0, 1, 3, 2, 5, 4],  # baab
         [0, 1, 3, 4, 2, 5]]  # bbaa

    Returns:
        Each inner list contains the permutation indices for symmetrization.
    """
    # TODO, this is a generalization of get_sym_rule_2 and get_sym_rule_3 in tensor_product1.py
    #  Can we merge them?

    all_perms = itertools.permutations(range(start_dim, start_dim + len(symmetry)))

    prefix = list(range(start_dim))
    unique_perms = []
    unique_perm_string = set()

    # Filter permutations based on the symmetry
    for perm in all_perms:
        perm_string = "".join(symmetry[i - start_dim] for i in perm)

        if perm_string not in unique_perm_string:
            unique_perms.append(prefix + list(perm))
            unique_perm_string.add(perm_string)

    return unique_perms


def get_permutations_2(m: int, num_delta: int, start_dim: int = 0) -> list[list[int]]:
    """
    Get the unique permutations of the tensor product of a symmetric tensor and deltas.

    For example, we know
    {U_rrss \delta_ij \delta_kl}
    = U_rrss \delta_ij \delta_kl
    + U_rsrs \delta_ij \delta_kl
    + U_rssr \delta_ij \delta_kl

    This is equivalent to
    1. First get V_ijkl = U_rrss \delta_ij \delta_kl
    2. Then permute V_ijkl to get V_ikjl and V_iklj
    3. Sum them up to get the result, i.e.
        {U_rrss \delta_ij \delta_kl} = V_ijkl + V_ikjl + V_iklj


    This function find the permutations of the indices in V.
    There are two types of symmetry to consider in the permutations:
    a. Minor symmetry: the symmetry of the two indices in each delta tensor.
       For example, V_ijkl = V_jikl = V_ijlk
    b. Major symmetry: the symmetry of indices between the deltas. For example,
       V_ijkl = V_klij

    In addition, we consider another symmetry:
    c. The symmetry of the remaining indices of the tensor, e.g. in U_rrst\delta_ij,
        the indices r and s are symmetric.

    Args:
        m: the rank of the symmetric tensor
        num_delta: the number of delta tensors to be contracted
        start_dim: the starting dimension to perform the operation. Dimensions before
            `start_dim` will not be used in the operation.

    Returns:
        Each inner list contains the permutation indices for symmetrization.
    """

    def get_canonical_form(ps: str, exclude: str) -> str:
        """
        Convert a permutation string to its canonical form, such that equivalent
        permutation strings have the same representation.

        This is based on first occurrence positions of each letter in the string.
        For example, `baba` and `fefe` are equivalent permutation strings, and
        both will be converted to `0101`.

        Args:
            ps: The permutation string to convert
            exclude: The letter to exclude from being changed, but keep its
                original value in the canonical form

        Returns:
            The canonical form of the permutation string
        """
        return "".join(c if c == exclude else str(ps.index(c)) for c in ps)

    num_remain = m - 2 * num_delta
    assert num_remain >= 0, "The number of remaining indices must be non-negative."

    # Construct the symmetry pattern
    # e.g., zzzzaabb, meaning we have 4 symmetric indices from the tensor and 2 pairs
    # of symmetric indices from two deltas.
    u_remain = "z" * num_remain
    delta = "".join(repeat_double_index(num_delta))
    symmetry = f"{u_remain}{delta}"

    all_perms = itertools.permutations(range(start_dim, start_dim + len(symmetry)))

    prefix = list(range(start_dim))
    unique_perms = []
    unique_canonical_forms = set()

    # Filter permutations based on contraction pattern
    for perm in all_perms:
        perm_string = "".join(symmetry[i - start_dim] for i in perm)

        # Use canonical form to identify equivalent patterns, dealing with both minor
        # symmetry and major symmetry.
        # The index 'z' is excluded to be changed, but keep its original value, because
        # it represents the remaining indices of the tensor, not associated with the
        # deltas.
        canonical_form = get_canonical_form(perm_string, exclude="z")

        if canonical_form not in unique_canonical_forms:
            unique_perms.append(prefix + list(perm))
            unique_canonical_forms.add(canonical_form)

    return unique_perms


def contract_with_epsilon(t: Tensor, indices: tuple[int, int]) -> Tensor:
    """
    Double contract the Levi-Civita tensor \epsilon_abc with a generic tensor t:

    \epsilon_abc T_...dbce... -> S_a...de...

    where the indices b and c are contracted away, and this results in a tensor of rank
    one less than the input tensor.

    Args:
        t: the tensor
        indices: the two positions of the indices in t to contract with the epsilon
    """

    letters = list(letter_index(t.ndim, start=3))
    letters[indices[0]] = "b"
    letters[indices[1]] = "c"
    t_rule_left = "".join(letters)
    t_rule_right = t_rule_left.replace("b", "").replace("c", "")

    rule = f"abc,{t_rule_left}->a{t_rule_right}"

    return torch.einsum(rule, eijk(), t)


# TODO, implement start_dim
def get_contraction_with_delta_rules(n: int, num_delta: int) -> list[str]:
    """
    Get all contraction rules of a generic tensor with deltas.

    T_...aijklb... delta_ij, delta_kl, ..., -> S_...ab...

    This considers symmetries between the deltas.
    For example, T_ijkl delta_ij delta_kl is the same as T_klij delta_ij delta_lk, and
    thus the two will be considered only once.

    Args:
        n: the rank of the tensor
        num_delta: the number of delta tensors to contract with

    Returns:
        A list of contraction rules, e.g. ['abc,bc->a', 'abc,ac->b', 'abc,ab->c'] for
        n=3 and num_delta=1.
    """
    # For example, for a rank-5 tensor
    #
    # S_mijkl = T_mrrss delta_ij delta_kl,
    # which corresponds to permutation (0, 1, 2, 3, 4).
    # 0, 1, 2, 3, 4 can be thought of as the positions of the indices in the tensor.
    # Then we see that, (0, 1, 2, 3, 4) means the first two indices are contracted
    # and the next two indices are contracted, ...
    # Then, we have:
    #
    # Permutation        Contraction
    # (0, 1, 2, 3, 4)     T_mrrss
    # (0, 1, 3, 2, 4)     T_mrsrs
    # ...
    # (1, 4, 2, 3, 0)     T_rsrsm
    #
    # The first rank-2*num_delta indices of t are not contracted, and the rest
    # 2*num_delta are contracted with deltas. In other words, the last 2*num_delta
    # indices are contracted with deltas.
    #

    perms = get_permutations_2(n, num_delta)

    t_indices = letter_index(n)

    start_idx = n - 2 * num_delta

    rules = []
    for p in perms:
        delta = [
            t_indices[p.index(start_idx + 2 * i)]
            + t_indices[p.index(start_idx + 2 * i + 1)]
            for i in range(num_delta)
        ]
        remaining = sorted(set(t_indices) - set("".join(delta)))

        r = f"{t_indices},{','.join(delta)}->{''.join(remaining)}"
        rules.append(r)

    return rules


if __name__ == "__main__":
    e = eijk()
    torch.manual_seed(0)
    # T = torch.arange(27).reshape(3, 3, 3).to(torch.float)
    T = torch.rand(3, 3, 3)

    out1 = torch.einsum("pij,ijk->pk", e, T)
    print(out1)

    out2 = torch.einsum("pij,jki->pk", e, T)
    print(out2)

    out3 = torch.einsum("pij,kij->pk", e, T)
    print(out3)

    print(out2 + out3)
    print(out2 + out3 + out1)

    x = get_permutations_2(6, 1)
    print(len(x))
