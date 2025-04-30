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
    """
    Interface to annotate ModuleList for TorchScript.

    Note, this should have exactly the same signature (including argument name
    `input` here) as the module it tries to annotate.

    See https://github.com/pytorch/pytorch/issues/68568
    """

    def forward(self, input: Tensor) -> Tensor:
        pass


class BufferDict(nn.Module):
    """A dictionary of buffers for PyTorch."""

    def __init__(self, d: dict[str, Tensor]):
        super().__init__()
        for k, v in d.items():
            self.register_buffer(k, v)

    def __getitem__(self, k: str) -> Tensor:
        return getattr(self, k)

    def __setitem__(self, k: str, v: Tensor) -> None:
        self.register_buffer(k, v)

    def __contains__(self, k: str) -> bool:
        return hasattr(self, k)

    def keys(self):
        return dict(self.named_buffers()).keys()

    def values(self):
        return dict(self.named_buffers()).values()

    def items(self):
        return self.named_buffers()
