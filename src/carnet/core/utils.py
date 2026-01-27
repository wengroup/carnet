from torch import Tensor


def check_shape(T: Tensor, n: int = 3) -> bool:
    """Check a tensor is of shape (n, n, ..., n)"""
    if T.ndim == 0:
        return True
    elif set(T.shape) != {n}:
        return False
    else:
        return True
