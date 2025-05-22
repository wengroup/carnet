import gzip
import json
from pathlib import Path

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


def json_dump(obj: dict, filename: Path, compress: bool = True) -> None:
    """Dump a dictionary to a json file.

    Args:
        obj: The dictionary to dump.
        filename: The path to the json file.
        compress: How to compress the file. If None, no compression is used. Options
            are: `gz` and `xz`.


    """
    if compress:
        filename = filename.with_suffix(filename.suffix + ".gz")
        with gzip.open(filename, "wt") as f:
            json.dump(obj, f)
    else:
        with open(filename, "w") as f:
            json.dump(obj, f)


def json_load(filename: Path) -> dict:
    """Load a json file into a dictionary.

    Args:
        filename: The path to the json file.

    Returns:
        The loaded dictionary.
    """
    if filename.suffix == ".gz":
        with gzip.open(filename, "rt") as f:
            return json.load(f)
    else:
        with open(filename, "r") as f:
            return json.load(f)
