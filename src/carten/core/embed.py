import torch
from natt.utils import dij, double_index, eijk, letter_index
from torch import Tensor


# TODO, whether this is needed? Can we simply embed using rank-T.ndim identity?
#  What is the difference and implication on rotations and tensor products?
def embed(T: Tensor, rank: int) -> Tensor:
    """
    Embed a natural tensor into a higher dimensional space.

    This is achieved by the below.
    1. Dot product a tensor with the epsilon tensor increased the rank by 1.
    2. Tensor product the tensor with the delta tensor increased the rank by 2.
    3. Tensor product multiple delta tensors to increase an even number of ranks.
    4. Tensor product multiple delta tensors and one epsilon tensors to increase an
       odd number of ranks.
    The operations do not change the rotation properties of the tensor.

    Reference:
    1. Section 3.2 of `IrreducibleCartesian Tensors`, Robert F. Snider, 2018,
       https://doi.org/10.1515/9783110564860

    Args:
        T: the natural tensor to embed
        rank: the rank of the higher dimensional space to embed the tensor

    Returns:
        The embedded tensor in the higher dimensional space.
    """
    if rank < T.ndim:
        raise ValueError("Rank must be greater than or equal to the tensor rank.")
    if rank == T.ndim:
        return T

    rank_diff = rank - T.ndim

    num_epsilon = rank_diff % 2
    num_delta = rank_diff // 2

    rule_left = letter_index(T.ndim)
    rule_right = letter_index(T.ndim)
    data = [T]

    if num_epsilon > 0:
        epsilon_index = rule_left[-1:] + letter_index(2, start=T.ndim)
        rule_left += "," + epsilon_index
        rule_right = rule_right[:-1] + epsilon_index[-2:]
        data += [eijk(device=T.device)]

    if num_delta > 0:
        delta_index = double_index(num_delta, start=T.ndim + 2)
        rule_left += "," + ",".join(delta_index)
        rule_right += "".join(delta_index)
        data += [dij(device=T.device)] * num_delta

    rule = rule_left + "->" + rule_right

    return torch.einsum(rule, *data)
