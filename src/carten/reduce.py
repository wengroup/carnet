"""Helper functions to decompose the special tensors or tensor product of two tensors
into natural tensors."""
import itertools
import string
import warnings

import torch
from torch import Tensor

from carten.natural_tensor import NaturalTensors
from carten.utils import dij, letter_index


def reduce_symmetric_tensor(U: Tensor, start_dim: int = 0) -> NaturalTensors:
    """
    Decompose a fully symmetric tensor into natural tensors.

    Args:
        U: a symmetric tensor
        start_dim: the starting dimension to treat U as a symmetric tensor. Dimensions
            before start_dim will be treated as batch dimensions.

    Returns:
        A NaturalTensors. Let n = U.ndim - start_dim; if n is even, there would be
        n/2 + 1 natural tensors of ranks n, n-2, ..., 2, 0. If n is odd, there would be
        (n+1)/2 natural tensors of rank n, n-2, ..., 3, 1.
    """

    def get_rule(indices: str, num_delta: int):
        delta_indices = ",".join(
            [indices[2 * i : 2 * i + 2] for i in range(0, num_delta)]
        )
        right = indices[num_delta * 2 :]
        return f"{delta_indices}, ...{indices} -> ...{right}"

    n = U.ndim - start_dim
    indices = letter_index(n)
    delta = dij()

    output = [remove_trace(U, start_dim=start_dim)]
    for i in range(1, n // 2 + 1):
        rule = get_rule(indices, i)
        data = [delta] * i + [U]
        symmetrized = torch.einsum(rule, data)
        traceless = remove_trace(symmetrized, start_dim=start_dim)
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


def symmetrize_and_remove_trace(t: Tensor, start_dim: int = 0) -> Tensor:
    """
    Symmetrize and remove the trace of a (generic) tensor.

    This only extracts the symmetric traceless part of the tensor at the same rank.
    The antisymmetric part is totally ignored, which can further be decomposed into
    symmetric and traceless tensors of lower ranks.

    Args:
        t: input tensor
        start_dim: the starting dimension to perform the operation.

    Returns:
        A symmetric traceless tensor of the same rank as the input tensor.
    """
    return remove_trace(symmetrize(t, start_dim), start_dim)


def symmetrize(t: Tensor, start_dim: int = 0) -> Tensor:
    """
    Fully symmetrize a tensor.

    Args:
        t: input tensor
        start_dim: the starting dimension from which to symmetrize the tensor.

    Reference:
        Eq 9 of: http://dx.doi.org/10.1080/00018737800101454
    """
    # TODO, benchmarking torch.einsum and torch.permute.
    rank = t.ndim - start_dim

    indices = letter_index(rank)
    perms = itertools.permutations(indices, len(indices))
    rules = [f"...{indices}->...{''.join(p)}" for p in perms]

    # TODO, is there anyway to avoid torch.stack? This creates a large tensor
    #  requiring a lot of memory.
    sym_t = torch.mean(torch.stack([torch.einsum(s, t) for s in rules]), dim=0)

    return sym_t


def remove_trace(t: Tensor, start_dim: int = 0):
    """
    Remove the trace of a symmetric tensors to get natural tensors.

    Data starting from `start_dim` will be considered as a natural tensor, and the
    leading dimensions will be considered separately. For example, if `start_dim = 2`
    and the tensor has a shape of [2, 4, 3, 3], there will be 2*4 natural tensors, and
    each will be processed separately.

    Args:
        t: a fully symmetric tensor
        start_dim: the starting dimension from which an input data is considered as
            the natural tensor.

    Give a symmetric tensor S_abc...q, the symmetric traceless part is given by:

    T_abc...q = S_abc...q
        - 1/(2n-1) * \sum_C1 (\delta_ab S_rrc...q)
        + 1/(2n-1)(2n-3) * \sum_C2 (\delta_ab\delta_cd S_rrss...q)
        - 1/(2n-1)(2n-3)(2n-5) * \sum_C3 (\delta_ab\delta_cd\delta_ef S_rrsstt...q)
        + ...

    \sum_C1, \sum_C2, \sum_C3, ... are sums over all combinations of indices.
    Specifically,
    - \sum_C1 is:
      \delta_ab S_rrc...q + \delta_ac S_rbr...q + ... \delta_pq S_abc...rr

    - \sum_C2 is:
      \delta_ab \delta_cd S_rrss...q + \delta_ab \delta_ce S_rrsds...q + ...

    - \sum_C3 is:
    \delta_ab\delta_cd\delta_ef S_rrsstt...q + \delta_ab\delta_cd\delta_eg S_rrsstft...q
    + ...


    For even n (the number of indices of the tensor), there will be n/2 terms C_i sums,
    where i = 1, 2, ..., n/2. For odd n, there will be (n-1)/2 terms C_i sums, where
    i = 1, 2, ..., (n-1)/2.

    These sums are formed by combination of indices. For example, for n = 5. With
    indices `abcde`,
    - \sum_C1 is:

        i1 = abcde
        d1 = Choose(i1, 2)

        We can use d1 to form the deltas.

    - sum_C2 is:
        i1 = abcde
        d1 = Choose(i1, 2)
        i2 = i1 - d1
        d2 = Choose(i2, 2)

        We then use d1 and d2 to form the deltas, with duplicates removed.

    References:
        Eq 10 of http://dx.doi.org/10.1080/00018737800101454

    Returns:
        A symmetric traceless tensor of the same rank as the input tensor.
    """

    rank = t.ndim - start_dim
    indices = letter_index(rank)
    delta_indices = get_unique_choose_two(indices)

    # TODO, note that t is fully symmetric, so, no matter which two indices we choose to
    #  contract, the result is the same. Therefore, we can choose any combinations of
    #  the two indices. And then permute the remaining indices to get the final result.
    #  # this may be faster than the current implementation?
    t_out = t
    factor = 1
    delta = dij()
    for i, indices in delta_indices.items():
        operand = [t] + [delta] * i

        # Note, start_dim is dealt with in the `get_contraction_rule_2` function, where
        # the tensor `t` is always contracted from the tailing dimensions.
        v = torch.sum(
            torch.stack(
                [torch.einsum(get_contraction_rule_2(d, i), operand) for d in indices]
            ),
            dim=0,
        )
        factor = -factor / (2 * rank - 2 * i + 1)

        t_out = t_out + factor * v

    return t_out


def get_unique_choose_two(
    indices: str, remove_duplicates: bool = True
) -> dict[int, list[list[str]]]:
    """
    Get all unique (set of) choosing two indices from a string of indices.

    The rest of indices (not chosen ones) will be appended to the end of each group.

    Args:
        indices: a string of indices with no repeat letter, e.g. "abcde".
        remove_duplicates: whether to remove duplicates. For exmaple  ['cd', 'ab'] is a
            duplicate of ['ab' 'cd'].

     Returns:
        A dict of all unique (set of) choosing two indices. The number of two-indices
        emelents in each group is the key of the dict, and it goes from 1 to n//2,
        where n is the length of the indices. The values are the corresponding
        two-indices combinations, and the remaining indices.

       Example:
        >>> get_unique_choose_two("abc")
        {1: [["ab", "c"], ["ac", "b"], ["bc", "a"]]},

        >>> get_unique_choose_two("abcde")
        {1: [["ab", "cde"], ["ac", "bde"], ["ad", "bce"], ["ae", "bcd"],
             ["bc", "ade"], ["bd", "ace"], ["be", "acd"],
             ["cd", "abe"], ["ce", "abd"],
             ["de", "abc"],
            ],
         2: [["ab", "cd", "e"],
             ["ab", "ce", "d"],
             ["ab", "de", "c"],
             ["ac", "bd", "e"],
             ["ac", "be", "d"],
             ["ac", "de", "b"],
             ["ad", "bc", "e"],
             ["ad", "be", "c"],
             ["ad", "ce", "b"],
             ["ae", "bc", "d"],
             ["ae", "bd", "c"],
             ["ae", "cd", "b"],
             ["bc", "de", "a"],
             ["bd", "ce", "a"],
             ["be", "cd", "a"],
             # Note, others like ["cd", "ab", "e"] will not appear because it is a
             # duplicate of ["ab", "cd", "e"].
            ]
        }
    """

    indices = "".join(sorted(indices))

    results = {0: [[indices]]}
    for i in range(1, len(indices) // 2 + 1):
        current = []
        for dr in results[i - 1]:
            done = dr[:-1]
            rest = dr[-1]
            chosen = itertools.combinations(rest, 2)

            for ch in chosen:
                ch = "".join(ch)
                if len(rest) > len(ch):
                    rest_rest = "".join(sorted(set(rest) - set(ch)))
                    current.append(done + [ch, rest_rest])
                else:
                    current.append(done + [ch])

        if remove_duplicates:
            # frozenset(x[:i]) selects the current done indices and make it a key.
            # Note, we cannot use frozenset(x) to use all because it can remove
            # non-duplicates. For example, consider the case with 4 indices, `abcd`,
            # and we choose a single pair of indices (i.e. i = 1). Then, we want
            # ["ab", "cd"], ["ac", "bd"], ["ad", "bc"], ["bc", "ad"], ["bd", "ac"],
            # ["cd", "ab"] as the results. However, if we use frozenset(x), then
            # ["ab", "cd"], ["ac", "bd"], ["ad", "bc"] will be removed.
            #
            # The original list is kept as the value to keep the order of the elements
            # in the list, so that we don't change them.
            current = {frozenset(x[:i]): x for x in current}
            unique = set(current.keys())
            results[i] = [current[k] for k in unique]
        else:
            results[i] = current

    # remove the first element, which is the original indices
    results.pop(0)

    return results


# TODO, this seems not used anywhere. Remove it.
def get_contraction_rule_1(indices: list[str], num: int) -> str:
    """
    Get the contraction rule from a list of indices.

    Args:
        indices: a list of indices
        num: the number of index to be contracted

    Example:
        >>> get_contraction_rule_1(["ab"], 1)
        "aa"
        >>> get_contraction_rule_1(["ab", "c"], 1)
        "aac->c"
        >>> get_contraction_rule_1(["ac", "b"], 1)
        "aba->b"
        >>> get_contraction_rule_1(["bd", "ac"], 1)
        "abcb->ac"
        >>> get_contraction_rule_1(["ac", "bd"], 2)
        "abab"
        >>> get_contraction_rule_1(["bd", "ace"], 1)
        "abcbe->ace"
        >>> get_contraction_rule_1(["ac", "bd", "e"], 2)
        "ababe->e"
    """

    # get sorted letters, assume each letter only appears once
    left = "".join(sorted("".join(indices)))

    for i, x in enumerate(indices):
        if i < num:
            idx = left.index(x[1])
            left = left.replace(left[idx], x[0])

    right = "".join(indices[num:])

    if len(right) == 0:
        return left
    else:
        return f"{left}->{right}"


def get_contraction_rule_2(indices: list[str], num: int) -> str:
    """
    Get the contraction rule from a list of indices, and keep the rank of tensor.

    The tensor will be contracted with `num` delta tensors: delta_zy, delta_xw,
    delta_vu... and the rank of the tensor will be kept.

    Args:
        indices: a list of indices. The indices serving two purposes: (1) the indices
            of the tensor, and (2) the position of the indices to be contracted. For
            example, if indices = ["ac", "b"] and num = 1, then the tensor will actually
            be a tensor with three indices T_abc, and the first and third indices will
            be contracted, signified by "ac".
        num: the number of index to be contracted

    Example:
        >>> get_contraction_rule_2(["ab"], 1)
        "...aa,zy->zy"
        >>> get_contraction_rule_2( ["ab", "c"], 1)
        "...aac,zy->zyc"
        >>> get_contraction_rule_2( ["ac", "b"], 1)
        "...aba,zy->zby"
        >>> get_contraction_rule_2(["bd", "ac"], 1)
        "...abcb,zy->azcy"
        >>> get_contraction_rule_2(["ac", "bd"], 2)
        "...abab,zy,xw->zxyw"
        >>> get_contraction_rule_2(["bd", "ace"], 1)
        "...abcbe,zy->azcye"
        >>> get_contraction_rule_2(["ac", "bd", "e"], 2)
        "...ababe,zy,xw->zxywe"
    """
    letters = "".join(reversed(string.ascii_lowercase))

    # get sorted letters, e.g. `abcde...`, assuming each letter only appears once
    left = "".join(sorted("".join(indices)))
    right = left

    appendix = ""
    for i, pair in enumerate(indices[:num]):
        idx0 = left.index(pair[0])
        idx1 = left.index(pair[1])
        left = left.replace(left[idx1], pair[0])
        new_letters = letters[i * 2 : (i + 1) * 2]

        appendix += "," + new_letters

        right = right.replace(right[idx0], new_letters[0]).replace(
            right[idx1], new_letters[1]
        )

    rule = f"...{left}{appendix}->{right}"

    return rule
