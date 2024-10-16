import itertools
import string

import torch
from torch import Tensor

from carten import SETTINGS
from carten.core.signature import Signature
from carten.core.utils import check_shape, check_symmetric_traceless, dij, letter_index


class NaturalTensors:
    """A sequence of natural tensors.

    By definition, a natural tensor is symmetric and traceless.

    For a generic tensor, `symmetrize_and_remove_trace()` can be used to convert it
    to a natural tensor of the same rank. Note, it will only extract the symmetric part
    of the tensor at the same rank. The antisymmetric part is ignored, which can be
    further decomposed into symmetric and traceless tensors of lower ranks.
    """

    def __init__(self, signature: Signature, data: Tensor = None):
        if data is not None and data.shape[-1] != signature.dim:
            raise ValueError(
                f"Signature `{signature}` and the dimension of data `{data.shape[-1]}` "
                f"do not match."
            )
        self._signature = signature

        # self._data is of shape [..., dim], where dim is the flattened dimension of the
        # natural tensor with multiplicity considered, i.e.
        # dim = sum_i mul_i * 3**rank_i, where mul_i is the multiplicity of the i-th
        # natural tensor, and rank_i is the rank of the i-th natural tensor.
        self._data = data

    @property
    def chunks(self):
        """The chunks of the natural tensors.

        Each chunk is a tensor of shape (..., dim), where dim is the dimension of the
        natural tensor with multiplicity considered.
        """
        i = 0
        for dim in self.chunk_dims:
            yield self._data[..., i : i + dim]
            i += dim

    @property
    def shaped_chunks(self):
        """The chunks of the natural tensors with the data reshaped.

        The is related to the `chunks` property. In the chunked data, the shape of each
        chunk is (..., dim), where dim is the flattened dimension of the natural tensor
        with multiplicity considered. Here, it is reshaped to (..., mul, 3,3,...,3),
        where `mul` is the multiplicity of the chunk, and 3,3,...,3 is the unflattened
        shape of the natural tensor. Note, for scalar, the unflattened shape is (1,).
        """
        i = 0
        for dim, (mul, rank) in zip(self.chunk_dims, self.signature):
            yield unflatten_tensor_dims(self._data[..., i : i + dim], mul, rank)
            i += dim

    def get_chunk(self, i: int) -> Tensor:
        """Get the i-th chunk of the natural tensors."""
        start = sum(self.chunk_dims[:i])
        end = start + self.chunk_dims[i]
        return self._data[..., start:end]

    def get_shaped_chunk(self, i: int) -> Tensor:
        """Get the i-th chunk of the natural tensors with the data reshaped."""
        start = sum(self.chunk_dims[:i])
        end = start + self.chunk_dims[i]
        return unflatten_tensor_dims(
            self._data[..., start:end], self.chunk_muls[i], self.chunk_ranks[i]
        )

    def simplify(self):
        """Simplify the signature and the data."""
        return NaturalTensors(self._signature.simplify(), self._data)

    def reorder(self):
        """Reorder the signature and the data."""
        ordered = self._signature.reorder()
        sig = ordered.signature
        data = self._data[..., ordered.data_perm]
        return NaturalTensors(sig, data)

    def regroup(self):
        """Regroup the signature and the data.

        This is equivalent to  reorder() + simplify().
        """
        ordered = self._signature.reorder()
        sig = ordered.signature.simplify()
        data = self._data[..., ordered.data_perm]
        return NaturalTensors(sig, data)

    @property
    def data(self) -> list[Tensor]:
        """The data of the natural tensors."""
        return self._data

    @property
    def shape(self) -> torch.Size:
        """The shape of the natural tensors."""
        return self._data.shape

    @property
    def leading_shape(self):
        """The (arbitrary) leading shape. It can be any shape, such as the batch dim."""
        return self._data.shape[:-1]

    @property
    def leading_dim(self):
        """The (arbitrary) leading dimension."""
        return len(self.leading_shape)

    @property
    def signature(self) -> Signature:
        """Signature of the tensors: their multiplicity and rank."""
        return self._signature

    @property
    def num_chunks(self) -> int:
        """Number of chunks."""
        return len(self.signature)

    @property
    def chunk_muls(self):
        """The multiplicity of each chunk."""
        return self._signature.chunk_muls

    @property
    def chunk_ranks(self) -> list[int]:
        """The ranks of the irreps tensors."""
        return self._signature.chunk_ranks

    @property
    def chunk_dims(self) -> list[int]:
        """The dimensions of each chunk."""
        return self._signature.chunk_dims

    @property
    def min_rank(self) -> int:
        """Minimum rank of the tensors."""
        return min(self.chunk_ranks)

    @property
    def max_rank(self) -> int:
        """Maximum rank of the tensors."""
        return max(self.chunk_ranks)

    @property
    def device(self):
        return self._data.device

    @property
    def dtype(self):
        return self._data.dtype

    # TODO, let check be True by default?
    @classmethod
    def from_sequence(
        cls, data: list[Tensor], start_dim: int = 0, check: bool = SETTINGS.DEBUG
    ):
        """
        Create a NaturalTensors from a sequence of tensors.

        This is the same as `from_chunks()` with each tensor as a chunk the multiplicity
        of each chunk is 1.

        Args:
            data: a list of tensors. The dimensions from `start_dim` to the last dim is
                regarded as the dimension of the natural tensor, and there can be any
                number of leading dimensions. The leading dimensions of all tensors must
                be the same.
            start_dim: the starting dimension from which an input data is considered as
                a natural tensor. For example, if start_dim = 2, then the first two dims
                of an input tensor will be regarded as batch dimensions, and the rest
                will be regarded as the dimensions of a natural tensor.
            check: whether to check the consistency of the input tensors.
        """

        if check:
            # TODO, fix: this below checking does not work. We should check each
            #  individual natural tensor in t.
            for i, t in enumerate(data):
                if not check_shape(t):
                    raise ValueError(f"Input tensor {i} is not a 3D tensor.")
                if not check_symmetric_traceless(t, atol=1e-5):
                    raise ValueError(f"Input tensor {i} is not a natural tensor.")

        if len(set(tuple(t.shape[:start_dim] for t in data))) != 1:
            raise ValueError(
                f"Input tensors have different shapes before start_dim `{start_dim}`."
            )

        ranks = [t.ndim - start_dim for t in data]
        signature = Signature.from_ranks(ranks)

        if start_dim == 0:
            leading_shape = [1]
        else:
            leading_shape = data[0].shape[:start_dim]
        data = torch.cat([t.reshape(*leading_shape, -1) for t in data], dim=-1)

        return cls(signature, data)

    @classmethod
    def from_chunks(
        cls, signature: Signature, data: list[Tensor], check: bool = SETTINGS.DEBUG
    ):
        """
        Create a NaturalTensors from a sequence of tensor chunks.

        Each chunk is a tensor of shape (..., dim), where dim is the flattened dimension
        of the natural tensor with multiplicity considered. Where `...` should be the
        same for all chunks, and `dim` can be different for different chunks.

        Args:
            signature: signature of the tensors.
            data: a list of input tensor chunks.
            check: whether to check the consistency of the input tensors.
        """

        if check:
            # the dimension of each chunk agrees with the corresponding signature
            for i, (mr, t) in enumerate(zip(signature, data)):
                if mr.mul * 3**mr.rank != t.shape[-1]:
                    raise ValueError(
                        f"Signature `{signature}` and the dimension of data "
                        f"`{t.shape[-1]}` do not match for chunk {i}."
                    )

            # all leading dimensions of the input tensors are the same
            if not len(set(tuple(t.shape[:-1] for t in data))) == 1:
                raise ValueError(
                    f"Input chunk tensors have different shapes before the last dim."
                )

        return cls(signature, torch.cat(data, dim=-1))

    @classmethod
    def from_shaped_chunks(
        cls, signature: Signature, data: list[Tensor], check: bool = SETTINGS.DEBUG
    ):
        """
        Create a NaturalTensors from a sequence of shaped tensor chunks.

        Each shaped chunk is a tensor of shape (*leading_shape, mul, *ending_shape),
        where `leading_shape` can be any shape, such as the batch dimensions. `mul` is
        the multiplicity. `ending_shape` is the shape of the natural tensor. For scalar,
        it should be (1,); for a vector, it should be (3,); for a general natural
        tensor, it should be (3,3,...,3) with the number of 3 equal to the rank of the
        natural tensor.

        Args:
            signature: signature of the tensors.
            data: a list of input tensor chunks.
            check: whether to check the consistency of the input tensors.
        """
        out = [
            flatten_tensor_dims(t, mul=mr.mul, rank=mr.rank, check=check)
            for mr, t in zip(signature, data)
        ]

        return cls(signature, torch.cat(out, dim=-1))

    def __getitem__(self, item):
        return self._data[item]

    def __len__(self):
        """Total number of chunked tensors."""
        return len(self._signature)

    def __str__(self):
        return self.signature.__str__() + "\n" + self._data.__str__()


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


def flatten_tensor_dims(
    t: Tensor, mul: int, rank: int, check: bool = SETTINGS.DEBUG
) -> Tensor:
    """
    Flatten the dimensions representing the natural tensor and the multiplicity.

    Flatten a tensor of shape (*leading_shape, mul, *ending_shape) to a chunked tensor
    of shape (*leading_shape, -1). `leading_shape` can be any shape, such as the batch
    dimensions. `mul` is the multiplicity. `ending_shape` is the shape of the natural
    tensor. For scalar, it should be (1,); for a vector, it should be (3,); for a
    general natural tensor, it should be (3,3,...,3) with the number of 3 equal to the
    rank of the natural tensor. `mul` and `*ending_shape` are flattened to a single
    dimension.

    Args:
        t: the chunked tensor to be flattened.
        mul: expected multiplicity of the input chunked tensor.
        rank: expected rank of the input chunked tensor.
        check: whether to check the shape of the input chunked tensor.

    Returns:
        A flattened chunked tensor.
    """
    if check:
        if rank == 0:
            m = t.shape[-rank - 2]
            shape = t.shape[-rank - 1 :]
        else:
            m = t.shape[-rank - 1]
            shape = t.shape[-rank:]

        if m != mul:
            raise ValueError(
                "The multiplicity of the tensor from the data and from the signature "
                "do not match."
            )

        if (rank == 0 and tuple(shape) != (1,)) or (rank > 0 and set(shape) != {3}):
            raise ValueError(
                f"The last {rank} dimensions of the input tensor should all be 3, "
                f"representing the shape of the natural tensor. But got "
                f"{t.shape[-rank :]} instead."
            )

    if rank == 0:
        leading_shape = t.shape[: -rank - 2]
    else:
        leading_shape = t.shape[: -rank - 1]

    return t.reshape(*leading_shape, -1)


def unflatten_tensor_dims(
    t: Tensor, mul: int, rank: int, check: bool = SETTINGS.DEBUG
) -> Tensor:
    """
    Unflatten the dimensions representing the natural tensor and the multiplicity.

    Unflatten a chunked tensor of shape (*leading_shape, dim) to a tensor of shape
    (*leading_shape, mul, *ending_shape). `leading_shape` can be any shape, such as the
    batch dimensions. `mul` is the multiplicity. `ending_shape` is the shape of the
    natural tensor. For scalar, it should be (1,); for a vector, it should be (3,);
    for a general natural tensor, it should be (3,3,...,3) with the number of 3 equal to
    the rank of the natural tensor. `mul` and `*ending_shape` are flattened to a single
    dimension.

    Args:
        t: the input chunked tensor to be unflattened.
        mul: expected multiplicity of the input chunked tensor.
        rank: expected rank of the input chunked tensor.
        check: whether to check the shape of the input chunked tensor.

    Returns:
        An unflattened chunked tensor.
    """
    if check:
        if t.shape[-1] != mul * 3**rank:
            raise ValueError(
                f"The last dimension of the input tensor {t.shape[-1]} does not match "
                f"the expected multiplicity {mul} and rank {rank}."
            )

    leading_shape = t.shape[:-1]

    if rank == 0:
        ending_shape = [1]
    else:
        ending_shape = [3] * rank

    return t.reshape(*leading_shape, mul, *ending_shape)
