import torch
from torch import Tensor

from carten.core.utils import dij, eijk


def tp_with_delta_epsilon(
    t: Tensor, num_delta: int, num_epsilon: int, rule: str
) -> Tensor:
    """
    Contract a tensor with multiple delta tensors and a triple contraction with
    epsilon tensor.

    Args:
        t: the tensor
        num_delta: the number of delta
        rule: the rule to contract

    Returns:
        the contracted tensor
    """
    d = dij(t.device)
    deltas = [d] * num_delta
    e = eijk(t.device)
    eps = [e] * num_epsilon

    return torch.einsum(rule, *eps, *deltas, t)
