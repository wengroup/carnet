import itertools
import shutil
from pathlib import Path

import lightning as L
import pandas as pd
import torch
from lightning import Trainer
from line_profiler import profile
from torch_geometric.loader.dataloader import DataLoader

from carten.data.dataset import Dataset
from carten.data.transform import ConsecutiveAtomType
from carten.model.pl.pl_tensor_model import AtomicTensorLitModule
from carten.model.pl.utils import (
    get_args,
    get_git_commit,
    instantiate_class,
    load_model,
)
from carten.model.tensor_model import AtomicTensorModel


def get_dataset(
    filename: Path, target_name: str, atomic_number: list[int], r_cut: float
):
    dataset = Dataset(
        filename=filename,
        target_names=(target_name, "atomic_selector"),
        r_cut=r_cut,
        transform=ConsecutiveAtomType(atomic_number),
        log=False,
    )

    return dataset


def get_dataloaders(
    target_name,
    target_signature,
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

    trainset = get_dataset(trainset_filename, target_name, atomic_number, r_cut)
    train_loader = DataLoader(trainset, batch_size=train_batch_size, shuffle=True)

    valset = get_dataset(valset_filename, target_name, atomic_number, r_cut)
    val_loader = DataLoader(valset, batch_size=val_batch_size, shuffle=False)

    testset = get_dataset(testset_filename, target_name, atomic_number, r_cut)
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


def update_model_configs(config: dict, dataset: Dataset) -> dict:
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
    output_signature = config["data"]["target_signature"]

    for name in ["num_atom_types", "r_cut", "target_signature"]:
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
    config["model"]["output_signature"] = output_signature
    config["model"]["target_shift"] = shift
    config["model"]["target_scale"] = scale
    config["model"]["num_average_neigh"] = num_average_neigh

    print(f"Updated model configs - `num_atom_types`: {num_atom_types}")
    print(f"Updated model configs - `r_cut`: {r_cut}")
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

    for name in ["target_name", "target_signature"]:
        if name in config["loss"]:
            raise ValueError(
                f"Parameter {name} already provided in the `data` section of the "
                "config file. Please remove it from the `loss` section."
            )

    # Check ranks in target_signature and loss.ratio are consistent
    assert set(target_signature.keys()) == set(
        config["loss"]["ratio"].keys()
    ), "target_signature and loss.ratio have inconsistent ranks"

    config["loss"]["target_name"] = target_name
    config["loss"]["target_signature"] = target_signature

    print(f"Updated loss configs - `target_name`: {target_name}")
    print(f"Updated loss configs - `target_signature`: {target_signature}")

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
    m = AtomicTensorModel(**model_hparams)

    # TODO, enable jit
    # m = torch.jit.script(m)

    # torch.compile cannot work up to pytorch v2.4.0, because we need double gradients
    # to compute forces. Below is the error:
    # torch.compile with autograd does not support double backwards
    # Check the issue: https://github.com/pytorch/pytorch/issues/91469
    #
    # m = torch.compile(m)
    # TODO, It should work for tensor predictions

    model = AtomicTensorLitModule(
        m,
        loss_hparams=loss_hparams,
        metrics_hparams=metrics_hparams,
        optimizer_hparams=optimizer_hparams,
        lr_scheduler_hparams=lr_scheduler_hparams,
        ema_hparams=ema_hparams,
        other_hparams=other_hparams,
    )

    return model


@profile
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

    # create new model
    if restore_checkpoint is None:
        config = update_model_configs(config, train_loader.dataset)
        config = update_loss_configs(config)
        config["git_commit"] = get_git_commit()

        model = get_model(
            config["model"],  # do not pop to pass to other_hparams to track with WandB
            loss_hparams=config.pop("loss"),
            metrics_hparams=config.pop("metrics"),
            optimizer_hparams=config.pop("optimizer"),
            lr_scheduler_hparams=config.pop("lr_scheduler"),
            ema_hparams=config.pop("ema"),
            other_hparams=config,
        )

    # load from checkpoint
    else:
        print(f"Loading model from checkpoint: {restore_checkpoint}")
        model = load_model(AtomicTensorLitModule, AtomicTensorModel, restore_checkpoint)
    print(model)

    # Train
    try:
        callbacks = instantiate_class(config["trainer"].pop("callbacks"))
    except KeyError:
        callbacks = None

    try:
        logger = instantiate_class(config["trainer"].pop("logger"))

        ## TODO, for DEBUG only, should be commented out
        ## log gradients, parameter histogram and model topology
        ## for test run with small max_epoch, you might need to set `log_freq` such that
        ## this is executed at least once
        # logger.watch(model, log="all", log_graph=False)
    except KeyError:
        logger = None

    trainer = Trainer(callbacks=callbacks, logger=logger, **config["trainer"])

    # Note, passing ckpt_path to trainer.fit() to restore epoch, optimizer state,
    # lr_scheduler state, etc.
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
    # Remove the processed data directory
    shutil.rmtree("./processed", ignore_errors=True)

    config_file = Path(__file__).parent / "configs" / "config_nmr_tensor.yaml"

    config = get_args(config_file)
    main(config)

    # Remove the processed data directory
    shutil.rmtree("./processed", ignore_errors=True)
