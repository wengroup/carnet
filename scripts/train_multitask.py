import itertools
import os
import shutil
from pathlib import Path
from pprint import pprint

import lightning as L
import pandas as pd
import torch
from lightning import Trainer
from torch_geometric.loader.dataloader import DataLoader

from carten.data.dataset import DatasetMultiTask
from carten.data.transform import ConsecutiveAtomType
from carten.model.multitask_model import MultiTaskModel
from carten.model.pl.pl_multitask_model import MultiTaskLitModule
from carten.model.pl.utils import (
    get_args,
    get_git_commit,
    instantiate_class,
    load_model,
)


def get_dataset(
    filename: Path, target_names: list[str], atomic_number: list[int], r_cut: float
):
    dataset = DatasetMultiTask(
        filename=filename,
        target_names=target_names,
        r_cut=r_cut,
        transform=ConsecutiveAtomType(atomic_number),
        log=False,
    )

    return dataset


def get_dataloaders(
    target_name,
    target_mode,
    target_signature,
    target_symmetry,
    atomic_number,
    r_cut,
    trainset_filename,
    valset_filename,
    testset_filename,
    train_batch_size,
    val_batch_size,
    test_batch_size,
):
    # TODO, check target in dataframe is consistent with target_signature

    names = []
    for nm in target_name:
        # For tensor properties
        if "tensor" in nm:
            names.append(nm + "_natural")  # Always include natural tensor
            if target_mode == "natural":
                pass
            elif target_mode in ["full", "voigt"]:
                names.append(nm + "_" + target_mode)
            else:
                raise ValueError(f"Unknown target mode: {target_mode}.")
        # For energy and forces
        else:
            names.append(nm)

    trainset = get_dataset(trainset_filename, names, atomic_number, r_cut)
    train_loader = DataLoader(
        trainset, batch_size=train_batch_size, shuffle=True, drop_last=True
    )

    valset = get_dataset(valset_filename, names, atomic_number, r_cut)
    val_loader = DataLoader(valset, batch_size=val_batch_size, shuffle=False)

    testset = get_dataset(testset_filename, names, atomic_number, r_cut)
    test_loader = DataLoader(testset, batch_size=test_batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


def update_data_configs(config: dict) -> dict:
    """Get atomic number from the train set, so we do not need to provide it in the
    config file"""

    atomic_number = config["data"].get("atomic_number", None)
    if atomic_number is None:
        filename = config["data"]["trainset_filename"]
        df = pd.read_json(filename)
        atomic_number = df["atomic_number"].to_list()
        unique_atomic_number = sorted(set(itertools.chain.from_iterable(atomic_number)))

        config["data"]["atomic_number"] = unique_atomic_number

        print(f"Updated data configs - `atomic_number`: {unique_atomic_number}")

    return config


def update_model_configs(config: dict, dataset: DatasetMultiTask) -> dict:
    """Update the model configs in the config file.

    A couple of internally determined parameters are added to the `model` section of
    the config file.

    Args:
        config: The entire config file.
        dataset: The dataset object.

    Returns:
        Update config dict.
    """
    # params already provided in the `data` section of the config file
    num_atom_types = len(config["data"]["atomic_number"])
    r_cut = config["data"]["r_cut"]
    target_name = config["data"]["target_name"]
    output_signature = config["data"]["target_signature"]

    for name in ["num_atom_types", "r_cut", "target_name", "target_signature"]:
        if name in config["model"]:
            raise ValueError(
                f"Parameter {name} already provided in the `data` section of the "
                "config file. Please remove it from the `model` section."
            )

    # params determined from dataset
    # If not provided or set to "auto" in the config file, use the values determined
    # from the dataset
    shift = config["model"].pop("target_shift", None)
    scale = config["model"].pop("target_scale", None)

    if (isinstance(shift, str) and shift.lower() == "auto") or (
        isinstance(scale, str) and scale.lower() == "auto"
    ):
        shift_tmp, scale_tmp = dataset.get_shift_and_scale_tensors()
        if isinstance(shift, str) and shift.lower() == "auto":
            shift = shift_tmp
        if isinstance(scale, str) and scale.lower() == "auto":
            scale = scale_tmp

    num_average_neigh = config["model"].pop("num_average_neigh", None)
    if num_average_neigh is None:
        num_average_neigh = "auto"
    if num_average_neigh.lower() == "auto":
        num_average_neigh = dataset.get_num_average_neigh()

    # update config file
    config["model"]["num_atom_types"] = num_atom_types
    config["model"]["r_cut"] = r_cut
    config["model"]["target_name"] = target_name
    config["model"]["output_signature"] = output_signature
    config["model"]["target_shift"] = shift
    config["model"]["target_scale"] = scale
    config["model"]["num_average_neigh"] = num_average_neigh

    print(f"Updated model configs - `num_atom_types`: {num_atom_types}")
    print(f"Updated model configs - `r_cut`: {r_cut}")
    print(f"Updated model configs - `target_name`: {target_name}")
    print(f"Updated model configs - `output_signature`: {output_signature}")
    print(f"Updated model configs - `target_shift`: {shift}")
    print(f"Updated model configs - `target_scale`: {scale}")
    print(f"Updated model configs - `num_average_neigh`: {num_average_neigh}")

    return config


def update_loss_configs(config: dict) -> dict:
    """
    Update loss configs from data config.

    This is to avoid redundancy in the config file, by providing the same parameters in
    both the `data` and `loss` sections of the config file.
    """
    target_name = config["data"]["target_name"]
    target_signature = config["data"]["target_signature"]
    target_mode = config["data"]["target_mode"]
    target_symmetry = config["data"].get("target_symmetry", None)

    for name in ["target_name", "target_mode", "target_signature", "target_symmetry"]:
        if name in config["loss"]:
            raise ValueError(
                f"Parameter {name} already provided in the `data` section of the "
                "config file. Please remove it from the `loss` section."
            )

    # Check ranks in target_signature and loss.ratio are consistent
    for k, v in target_signature.items():
        if k not in config["loss"]["ratio"]:
            raise ValueError(f"{k} not provided in loss.ratio")

    config["loss"]["target_name"] = target_name
    config["loss"]["target_mode"] = target_mode
    config["loss"]["target_signature"] = target_signature
    config["loss"]["target_symmetry"] = target_symmetry

    print(f"Updated loss configs - `target_name`: {target_name}")
    print(f"Updated loss configs - `target_mode`: {target_mode}")
    print(f"Updated loss configs - `target_signature`: {target_signature}")
    print(f"Updated loss configs - `target_symmetry`: {target_symmetry}")

    return config


def get_model(
    model_hparams: dict,
    loss_hparams=None,
    metrics_hparams=None,
    optimizer_hparams=None,
    lr_scheduler_hparams=None,
    ema_hparams=None,
    other_hparams: dict = None,
):
    m = MultiTaskModel(**model_hparams)

    # TODO, enable jit
    # m = torch.jit.script(m)

    # torch.compile cannot work up to pytorch v2.4.0, because we need double gradients
    # to compute forces. Below is the error:
    # torch.compile with autograd does not support double backwards
    # Check the issue: https://github.com/pytorch/pytorch/issues/91469
    #
    # m = torch.compile(m)
    # TODO, It should work for tensor predictions

    model = MultiTaskLitModule(
        m,
        loss_hparams=loss_hparams,
        metrics_hparams=metrics_hparams,
        optimizer_hparams=optimizer_hparams,
        lr_scheduler_hparams=lr_scheduler_hparams,
        ema_hparams=ema_hparams,
        other_hparams=other_hparams,
    )

    return model


def main(config: dict):
    L.seed_everything(config["seed_everything"])

    # Set default dtype
    dtype = config.get("default_dtype", "float32")
    torch.set_default_dtype(getattr(torch, dtype))

    # Load data
    config = update_data_configs(config)
    train_loader, val_loader, test_loader = get_dataloaders(**config["data"])

    # Get model
    restore_checkpoint = config.pop("restore_checkpoint")

    # Update configs
    config = update_model_configs(config, train_loader.dataset)
    config = update_loss_configs(config)
    config["git_commit"] = get_git_commit()

    # Create new model
    if restore_checkpoint is None:
        model = get_model(
            config["model"],  # do not pop to pass to other_hparams to track with WandB
            loss_hparams=config.pop("loss"),
            metrics_hparams=config.pop("metrics"),
            optimizer_hparams=config.pop("optimizer"),
            lr_scheduler_hparams=config.pop("lr_scheduler"),
            ema_hparams=config.pop("ema"),
            other_hparams=config,
        )
    # Load from checkpoint
    else:
        print(f"Loading model from checkpoint: {restore_checkpoint}")

        # Pass the model hyperparameters to override the ones saved in the checkpoint.
        # This becomes useful when changing the way to train the model, e.g. using a
        # different loss weight.
        # Note, optimizer and lr_scheduler, will not be effective although they are
        # passed here, as they will be restored from the checkpoint below with
        # trainer.fit(ckpt_path=restore_checkpoint).
        names = {
            "loss": "loss_hparams",
            "metrics": "metrics_hparams",
            # "optimizer": "optimizer_hparams",
            # "lr_scheduler": "lr_scheduler_hparams",
            "ema": "ema_hparams",
        }
        overrides = {v: config.pop(k) for k, v in names.items()}

        model = load_model(
            MultiTaskLitModule,
            MultiTaskModel,
            restore_checkpoint,
            overrides=overrides,
        )

    # Train
    try:
        callbacks = instantiate_class(config["trainer"].pop("callbacks"))
    except KeyError:
        callbacks = None

    try:
        logger = instantiate_class(config["trainer"].pop("logger"))

        ## TODO, for DEBUG only, should be commented out
        ## log gradients, parameter histogram and model topology
        ## For test run with small max_epoch, you might need to set `log_freq` to a
        ## smaller value (default is 100) so that this is executed at least once.
        # logger.watch(model, log="all", log_graph=False, log_freq=1)
    except KeyError:
        logger = None

    trainer = Trainer(callbacks=callbacks, logger=logger, **config["trainer"])

    # Note, passing ckpt_path to trainer.fit() to restore epoch, optimizer state,
    # lr_scheduler state, etc.
    # See: https://lightning.ai/docs/pytorch/1.6.0/common/checkpointing.html#restoring-training-state
    # Note, in a restoring training, if, e.g., lr_scheduler hyperparameters are changed
    # and new values are provided in `config`, they won't be updated in the training
    # process, since the below `fit` method has `ckpt_path` as an argument, which will
    # override the hyperparameters from the config file (set in the above line).
    trainer.fit(
        model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
        ckpt_path=restore_checkpoint,
    )

    # Save the last epoch model
    # The behavior of  `save_last` in ModelCheckpoint callback is buggy; save manually
    trainer.save_checkpoint("./last_epoch.ckpt")

    # Test results on the best model determined by the validation set
    out = trainer.test(ckpt_path="best", dataloaders=test_loader)
    print("Best model test results:", out)
    print(f"Best checkpoint path: {trainer.checkpoint_callback.best_model_path}")

    # Validation results on best model
    out = trainer.validate(ckpt_path="best", dataloaders=val_loader)
    print("Best model val results:", out)

    # Val/test results on the last epoch model
    # Depending on the settings, this can be
    # - the last epoch of regular model
    # - the EMA model
    # - the SWA model
    # out = trainer.validate(ckpt_path="./last_epoch.ckpt", dataloaders=val_loader)
    # print("Last epoch results on val set:", out)
    #
    # out = trainer.test(ckpt_path="./last_epoch.ckpt", dataloaders=test_loader)
    # print("Last epoch results on test set:", out)


if __name__ == "__main__":

    config_file = Path(__file__).parent / "configs" / "config_multitask.yaml"
    config = get_args(config_file)
    pprint(config)

    # Set wandb proxy
    os.environ["WANDB_BASE_URL"] = config.pop("wandb_base_url")

    ## If the above does not work, use the below
    # import swanlab
    #
    # # Hijack WandB to use SwanLab
    # # This makes WandB to run in `offline` mode
    # swanlab.sync_wandb(wandb_run=False)

    main(config)

    # Remove the processed data directory to save space
    shutil.rmtree("./processed", ignore_errors=True)
