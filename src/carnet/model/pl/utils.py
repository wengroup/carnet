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
    params_load_mode: str = "separate",
    reset_ema_step: bool = False,
):
    """
    Load the lightning module from checkpoint.

    This will restore the lightning module. It first creates a new instance of
    `lit_model_cls` using the same hyperparameters saved in the checkpoint (override
    them if `override` is provided, and then loads model parameters (including ema)
    from the state dict of the checkpoint. So, if anything in `overrides` (direct or
    indirect) is saved in state dict (e.g. via register_buffer), `overrides` will not
    take effect.

    Currently, the lighting module (e.g. InteratomicPotentialLitModule) takes as
    arguments the below hyperparameters:
    - model and ema model. Both are created and parameters loaded from state dict.
    - `loss` and `metrics`. They will be created, but no internal states to load.
    - `ema`. The ema instance will be created and its internal states (`initted` and
       `step`) will be restored from the state dict.
    - `optimizer` and `lr_scheduler`. They will be created; however, this only creates
       the optimizer and lr_scheduler instances, but not load their states
       (e.g. momentum, learning rate, etc) from the checkpoint. These can be done by
       trainer.fit(ckpt_path=checkpoint) if you want to restore the training state.
       Also trainer.fit(ckpt_path=checkpoint) will restore the epoch, global step, etc.

    Note, if trainer.fit(ckpt_path=checkpoint) is called after this function, settings
    like `overrides` and `params_load_mode` will not be effective, as they will be
    overwritten by the checkpoint loaded by the trainer.

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
        params_load_mode: The checkpoint stores two set of parameters: "running"
            parameters and "ema" parameters. This specifies how to load the parameters
            into the model and the ema model.
            - "separate": load the "running" parameters into the model and the "ema"
               parameters into the ema model. This is good for resuming training.
            - "running": load the "running" parameters into both the model and the ema
               model.
            - "ema": load the "ema" parameters into both the model and the ema model.
              This is good for evaluation or inference, as the "ema" parameters usually
              have better performance than the "running" parameters. This is typically
              better than "running" for finetuning as well, as the "ema" parameters are
              usually more stable than the "running".
        reset_ema_step: whether to reset the internal ema step to 0 after loading the
            checkpoint, which means, such that ema starts like from scratch. This can
            be useful for finetuning.
    """

    # Create a pure PyTorch `model_cls` model
    # Note, we cannot simply do load_from_checkpoint(checkpoint) to load the model.
    # We have to first create a pure `model_cls` model and pass it to
    # load_from_checkpoint. This is because in `lit_model_cls`, we pass `model` as a
    # positional argument to the `__init__` method, and it is not saved in the
    # checkpoint.
    # For more, see the `encoder` example here:
    # https://lightning.ai/docs/pytorch/stable/common/checkpointing_basic.html#initialize-with-other-parameters
    #
    d = torch.load(checkpoint, map_location=map_location, weights_only=True)
    model_hparams = d["hyper_parameters"]["other_hparams"]["model"]

    # Update model_hparams from the checkpoint with values provided in `override`.
    # Of course, most model_hparams (e.g. channel size) should never be overridden,
    # but some (e.g. atomic energy shift) can be overridden, e.g.,  for finetuning.
    overrides = overrides or {}
    if overrides:
        model_hparams_config = overrides["other_hparams"]["model"]
        if model_hparams_config:
            for k, v in model_hparams_config.items():
                model_hparams[k] = v

            # Add model_hparams back to other_hparams, to be saved to checkpoint file
            overrides["other_hparams"]["model"] = model_hparams

    # Create model
    model = model_cls(**model_hparams)

    # Create lightning module
    lit_model = lit_model_cls.load_from_checkpoint(
        checkpoint, map_location=map_location, model=model, **overrides
    )

    # How model params and ema params are loaded
    if params_load_mode == "separate":
        # By default, lit_model_cls.load_from_checkpoint loads separately
        pass
    elif params_load_mode == "running":
        lit_model.ema.copy_params_from_model_to_ema()
    elif params_load_mode == "ema":
        lit_model.ema.copy_params_from_ema_to_model()
    else:
        supported = ["separate", "running", "ema"]
        raise ValueError(
            f"Unsupported params_load_mode: {params_load_mode}. Options: {supported}."
        )

    if reset_ema_step:
        # Set to 0
        # `step` is a tensor, so should not use `lit_model.ema.step = 0`, which sets
        # step to an int.
        lit_model.ema.step = lit_model.ema.step - lit_model.ema.step

    return lit_model


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
