import subprocess
from pathlib import Path

import torch
import yaml
from lightning.pytorch.cli import instantiate_class as lit_instantiate_class

from carten.training.lit_model import LitModel


def instantiate_class(d: dict | list[dict]):
    """Instantiate one or a list of LightningModule classes from a dictionary."""
    args = tuple()  # no positional args
    if isinstance(d, dict):
        return lit_instantiate_class(args, d)
    elif isinstance(d, list):
        return [lit_instantiate_class(args, x) for x in d]
    else:
        raise ValueError(f"Cannot instantiate class from {d}")


def get_args(path: Path):
    """Get the arguments from the config file."""
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    return config


def load_model(model_cls, checkpoint: Path, map_location: str = None):
    """
    Load the model from checkpoint.

    Args:
        model_cls: the PyTorch model class
    """
    # create the model to load, using the same hyperparameters saved in the checkpoint
    d = torch.load(checkpoint, map_location=map_location)

    # dtype
    dtype = d["hyper_parameters"]["other_hparams"]["default_dtype"]
    torch.set_default_dtype(getattr(torch, dtype))

    # create model
    model_hparams = d["hyper_parameters"]["other_hparams"]["model"]
    model = model_cls(**model_hparams)

    # create the lit model
    # `model` is passed here to overwrite default model in the checkpoint. The reason
    # we do this is because LitModel can potentially work with different model_cls.
    # Although model is provided, its state_dict will be updated from the checkpoint.
    # note, this will only restore model parameters, not the epoch, optimizer state,
    # lr_scheduler state, etc.
    lit_model = LitModel.load_from_checkpoint(
        checkpoint, map_location=map_location, model=model
    )

    return lit_model


def compile_model(lit_model):
    """compile a model using torch script."""
    model = lit_model.model
    compiled_model = torch.compile(model)
    return compiled_model


def get_git_commit(
    repo_path: Path = None, filename: str | None = "git_commit.txt"
) -> str:
    """
    Get the latest git commit info of a git repository.

    Args:
        repo_path: path to the repo
        filename: if not None, write the commit info to this file

    Returns:
        latest commit info
    """
    if repo_path is None:
        import carten

        repo_path = Path(carten.__file__).parents[1]

    output = subprocess.check_output(["git", "log"], cwd=repo_path)
    output = output.decode("utf-8").split("\n")[:6]
    latest_commit = "\n".join(output)

    if filename is not None:
        with open(filename, "w") as f:
            f.write(latest_commit)

    return latest_commit


if __name__ == "__main__":
    from carten.model.carten import CARTEN

    lit_model = load_model(
        CARTEN,
        "/Users/mjwen.admin/Packages/camp/scripts/last_epoch.ckpt",
    )

    compile_model(lit_model)
