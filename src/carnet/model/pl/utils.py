import subprocess
from pathlib import Path

import torch
import yaml
from lightning.pytorch.cli import instantiate_class as lit_instantiate_class


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


def load_model(
    lit_model_cls,
    model_cls,
    checkpoint: Path,
    map_location: str = None,
    overrides: dict | None = None,
):
    """
    Load the lightning module from checkpoint.

    This will fully restore the lightning module: first create a new instance of
    `lit_model_cls` using the same hyperparameters saved in the checkpoint, and then
    load the model parameters in the state dict of the checkpoint.

    Note this will not load parameters related to training, e.g. epoch, optimizer state,
    lr_scheduler state, etc, although they are also saved in the checkpoint.
    If you want to restore the training state, they can be loaded via
    trainer.fit(ckpt_path=checkpoint).

    Args:
        lit_model_cls: the Lightning model class, e.g. `InteratomicPotentialLitModule`
        model_cls: the pure model class, e.g. `InteratomicPotential`
        checkpoint: path to the checkpoint
        map_location: device to load the model to
        overrides: additional hyperparameters to override these saved in the checkpoint.
            They will be used to instantiate the lightning module, instead of the ones
            saved in the checkpoint. Accepts a dictionary of hyperparameters, each
            should be of the same format as defined in the __init__ method of the
            `lit_model_cls`.
    """

    # Create a pure PyTorch `model_cls` model
    # Note, we cannot simply do load_from_checkpoint(checkpoint) to load the model.
    # We have to first create a pure `model_cls` model and pass it to
    # load_from_checkpoint. This is because in `lit_model_cls`, we pass `model` as a
    # positional argument to the `__init__` method, and it is not saved in the
    # checkpoint.
    # For more, see the `encoder` example here:
    # https://lightning.ai/docs/pytorch/stable/common/checkpointing_basic.html#initialize-with-other-parameters
    d = torch.load(checkpoint, map_location=map_location, weights_only=True)
    model_hparams = d["hyper_parameters"]["other_hparams"]["model"]
    model = model_cls(**model_hparams)

    # Create lightning module
    overrides = overrides or {}
    lit_model = lit_model_cls.load_from_checkpoint(
        checkpoint, map_location=map_location, model=model, **overrides
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
        import carnet

        repo_path = Path(carnet.__file__).parents[1]

    output = subprocess.check_output(["git", "log"], cwd=repo_path)
    output = output.decode("utf-8").split("\n")[:6]
    latest_commit = "\n".join(output)

    if filename is not None:
        with open(filename, "w") as f:
            f.write(latest_commit)

    return latest_commit


def update_checkpoint(
    ckpt_path: Path, config: dict, output_path: Path = None, map_location: str = None
):
    """
    Update the checkpoint file to modify callback states etc.

    Args:
        ckpt_path: path to the checkpoint file
        config: the config dictionary
        output_path: path to save the updated checkpoint file
        map_location: device to load the model to
    """
    d = torch.load(ckpt_path, map_location=map_location)

    def update_dict(d: dict, config: dict, indent=""):
        """Update the dictionary with the config."""
        for k, v in config.items():
            if isinstance(v, dict):
                print(indent, k)
                indent += "  "
                d[k] = update_dict(d.get(k, {}), v, indent)
            else:
                print(f"{indent}{k}: {v} (Updated from {d[k]})")
                d[k] = v
        return d

    # update the hyper_parameters
    d = update_dict(d, config)

    output_path = output_path or ckpt_path
    torch.save(d, output_path)


if __name__ == "__main__":
    # from carnet.model.ip import InteratomicPotential
    # from carnet.model.pl.pl_ip import InteratomicPotentialLitModule
    #
    # lit_model = load_model(
    #     InteratomicPotentialLitModule,
    #     InteratomicPotential,
    #     "/Users/mjwen/Packages/carnet/scripts/last_epoch.ckpt",
    # )
    #
    # compile_model(lit_model)

    update_checkpoint(
        "/Users/mjwen/Packages/carnet/scripts/last_epoch.ckpt",
        {
            "callbacks": {
                "EarlyStopping{'monitor': 'val_ema/mae_e', 'mode': 'min'}": {
                    "patience": 500
                }
            }
        },
    )
