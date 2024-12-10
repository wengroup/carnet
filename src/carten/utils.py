import torch
from torch import Tensor, nn


def time_it(func, *args, **kwargs):
    """Time a function."""
    import time

    num_runs = 1

    times = []
    for _ in range(num_runs):
        start = time.time()
        out = func(*args, **kwargs)
        end = time.time()
        times.append(end - start)

    avg = sum(times) / num_runs
    print(f"Running {func.__name__} for {num_runs} times. Average time: {avg:.6e} s")

    return out


@torch.jit.interface
class JITInterface(nn.Module):
    """Note, this should have exactly the same signature (including argument name
    `input` here) as The module it tries to annotate.

    See https://github.com/pytorch/pytorch/issues/68568
    """

    def forward(self, input: Tensor) -> Tensor:
        pass
