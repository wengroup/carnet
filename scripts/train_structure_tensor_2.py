import itertools
import shutil
from pathlib import Path

import lightning as L
import pandas as pd
import swanlab
import torch
from line_profiler import profile
from torch_geometric.loader.dataloader import DataLoader

from carten.data.dataset import DatasetTensor
from carten.data.transform import ConsecutiveAtomType
from carten.model.pl.trainer import StructureTensorPyTorchTrainer
from carten.model.pl.utils import get_args, get_git_commit
from carten.model.tensor_model import StructureTensorModel


def get_dataset(
    filename: Path, target_names: list[str], atomic_number: list[int], r_cut: float
):
    dataset = DatasetTensor(
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

    names = [target_name + "_natural"]  # Always add natural target
    if target_mode == "natural":
        pass
    elif target_mode in ["full", "voigt"]:
        names.append(target_name + "_" + target_mode)
    else:
        raise ValueError(f"Unknown target mode: {target_mode}.")

    trainset = get_dataset(trainset_filename, names, atomic_number, r_cut)
    train_loader = DataLoader(trainset, batch_size=train_batch_size, shuffle=True)

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


def update_model_configs(config: dict, dataset: DatasetTensor) -> dict:
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
    target_mode = config["data"]["target_mode"]
    target_symmetry = config["data"].get("target_symmetry", None)

    for name in ["target_name", "target_mode", "target_signature", "target_symmetry"]:
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
    config["loss"]["target_mode"] = target_mode
    config["loss"]["target_signature"] = target_signature
    config["loss"]["target_symmetry"] = target_symmetry

    print(f"Updated loss configs - `target_name`: {target_name}")
    print(f"Updated loss configs - `target_mode`: {target_mode}")
    print(f"Updated loss configs - `target_signature`: {target_signature}")
    print(f"Updated loss configs - `target_symmetry`: {target_symmetry}")

    return config


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

        model = StructureTensorModel(**config["model"])

    # Load from checkpoint
    else:
        print(f"Loading model from checkpoint: {restore_checkpoint}")
        raise NotImplementedError

    print(model)

    trainer = StructureTensorPyTorchTrainer(
        model=model,
        loss_hparams=config.pop("loss"),
        metrics_hparams=config.pop("metrics"),
        optimizer_hparams=config.pop("optimizer"),
        lr_scheduler_hparams=config.pop("lr_scheduler"),
        ema_hparams=config.pop("ema"),
        other_hparams=config,
    )

    # Note, passing ckpt_path to trainer.fit() to restore epoch, optimizer state,
    # lr_scheduler state, etc.
    trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        max_epochs=2,
        checkpoint_dir="./checkpoints",
    )

    trainer.test(test_loader)

    # # Save the last epoch model
    # # The behavior of  `save_last` in ModelCheckpoint callback is buggy; save manually
    # trainer.save_checkpoint("./last_epoch.ckpt")
    #
    # # Test results on the best model determined by the validation set
    # out = trainer.test(ckpt_path="best", dataloaders=test_loader)
    # print("Best model test results:", out)
    # print(f"Best checkpoint path: {trainer.checkpoint_callback.best_model_path}")
    #
    # # Validation results on best model
    # out = trainer.validate(ckpt_path="best", dataloaders=val_loader)
    # print("Best model val results:", out)

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

    # Hijack WandB to use SwanLab
    # This makes WandB to run in `offline` mode
    swanlab.sync_wandb(wandb_run=False)

    # Remove the processed data directory
    shutil.rmtree("./processed", ignore_errors=True)

    # config_file = Path(__file__).parent / "configs" / "config_dielectric_tensor.yaml"
    config_file = Path(__file__).parent / "configs" / "config_elastic_tensor.yaml"

    config = get_args(config_file)
    main(config)

    # Remove the processed data directory
    shutil.rmtree("./processed", ignore_errors=True)
