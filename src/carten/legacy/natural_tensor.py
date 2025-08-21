import torch
from natt.utils import is_symmetric_traceless
from torch import Tensor

from carten import SETTINGS
from carten.core.utils import check_shape
from carten.legacy.signature import Signature


# TODO, this is a bit too complex, and not used for building the models currently,
#  because we require tensors of all ranks to have the same multiplicity.
#  We might want to use it if relax this constraint in the future.
#  Honestly, we might want to do it to speed up the calculations and reduce the
#  memory footprint: we can use smaller multiplicity for higher rank tensors.
class NaturalTensors:
    """A sequence of natural tensors.

    A natural tensor is symmetric and traceless.
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
        for mul, rank in self.signature:
            dim = mul * 3**rank
            yield unflatten_tensor_dims(self._data[..., i : i + dim], mul, rank)
            i += dim

    def get_chunk(self, i: int) -> Tensor:
        """Get the i-th chunk of the natural tensors."""
        dims = self._signature.chunk_dims
        start = sum(dims[:i])
        end = start + dims[i]
        return self._data[..., start:end]

    def get_shaped_chunk(self, i: int) -> Tensor:
        """Get the i-th chunk of the natural tensors with the data reshaped."""
        return unflatten_tensor_dims(
            self.get_chunk(i),
            self._signature.get_chunk_mul(i),
            self._signature.get_chunk_rank(i),
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
            # TODO fix: this below checking does not work. We should check each
            #  individual natural tensor in t.
            for i, t in enumerate(data):
                if not check_shape(t):
                    raise ValueError(f"Input tensor {i} is not a 3D tensor.")
                if not is_symmetric_traceless(t, atol=1e-5):
                    raise ValueError(f"Input tensor {i} is not a natural tensor.")

        if len(set(tuple(t.shape[:start_dim] for t in data))) != 1:
            raise ValueError(
                f"Input tensors have different shapes before start_dim `{start_dim}`."
            )

        ranks = [t.ndim - start_dim for t in data]
        signature = Signature.from_ranks(ranks)

        if start_dim == 0:
            # No batch dimensions, but we create one for it
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
            m = t.shape[-2]
            shape = t.shape[-1:]
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
        leading_shape = t.shape[:-2]
    else:
        leading_shape = t.shape[: -rank - 1]

    return t.view(*leading_shape, -1)


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

    return t.view(*leading_shape, mul, *ending_shape)
