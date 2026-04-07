import copy
import itertools
import os
import shutil
import warnings
from pathlib import Path
from pprint import pprint

import lightning as L
import pandas as pd
import torch
from lightning import Trainer
from torch_geometric.loader.dataloader import DataLoader

from carnet.data.dataset import DatasetIP
from carnet.data.transform import ConsecutiveAtomType
from carnet.model.ip import InteratomicPotential
from carnet.model.pl.pl_ip import InteratomicPotentialLitModule
from carnet.model.pl.utils import (
    get_args,
    get_git_commit,
    instantiate_class,
    load_model,
)
from carnet.utils import timer


def get_dataset(
    filename: Path, target_names: list[str], atomic_number: list[int], r_cut: float
):
    dataset = DatasetIP(
        filename=filename,
        target_names=target_names,
        r_cut=r_cut,
        transform=ConsecutiveAtomType(atomic_number),
        log=False,
    )

    return dataset


def get_dataloaders(
    atomic_number,
    r_cut,
    trainset_filename,
    valset_filename,
    testset_filename,
    train_batch_size,
    val_batch_size,
    test_batch_size,
    target_names=("energy", "forces"),
):

    trainset = get_dataset(trainset_filename, target_names, atomic_number, r_cut)
    train_loader = DataLoader(
        trainset, batch_size=train_batch_size, shuffle=True, drop_last=True
    )

    valset = get_dataset(valset_filename, target_names, atomic_number, r_cut)
    val_loader = DataLoader(valset, batch_size=val_batch_size, shuffle=False)

    testset = get_dataset(testset_filename, target_names, atomic_number, r_cut)
    test_loader = DataLoader(testset, batch_size=test_batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


def update_data_configs(config_ckpt: dict, config: dict) -> dict:
    """
    Update data configs.

    The below params can be different from values in the pretrained model:
    - r_cut

    All others should be exactly the same as values in the pretrained model. In
    particular, pay attention to:
    - atomic_number
    """

    # Check params that can be different
    k = "r_cut"
    ckpt_val = config_ckpt["data"][k]
    if k in config["data"]:
        #
        # If a param in the config is different from that in the checkpoint, use
        # value from the config, but issue a warning. This is legitimate, but we
        # hope the user know it.
        if config["data"][k] != ckpt_val:
            warnings.warn(
                f"Inconsistent `{k}`. value in the pretrained model is {ckpt_val}, "
                f"whereas value from the config is {config['data'][k]}."
            )
    else:
        config["data"][k] = ckpt_val
        print(f"Updated data configs - `{k}`: {ckpt_val}")

    # Set params that should be the same
    k = "atomic_number"
    ckpt_val = config_ckpt["data"][k]
    if k in config["data"]:
        raise ValueError(
            f"`{k}` should not be provided. Its value will be determined from the "
            f"pretrained model. Please remove it from the `data` section of the "
            f"config file."
        )
    else:
        # Ensure the atomic number from the dataset is a subset of those from the
        # checkpoint. If not, the checkpoint model is not pretrained on the atomic
        # number in the dataset. Then finetuning may not work well.
        filename = config["data"]["trainset_filename"]
        df = pd.read_json(filename)
        dataset_atomic_number = df[k].to_list()
        dataset_atomic_number = set(
            itertools.chain.from_iterable(dataset_atomic_number)
        )

        if not dataset_atomic_number.issubset(set(ckpt_val)):
            raise ValueError(
                "The pretrained model supports the following atomic numbers: "
                f"{ckpt_val}; while the atomic numbers in the dataset "
                f"are {sorted(dataset_atomic_number)}."
            )

        config["data"][k] = ckpt_val
        print(f"Updated data configs - `{k}`: {ckpt_val}")

    return config


def update_model_configs(config_ckpt: dict, config: dict, dataset: DatasetIP) -> dict:
    """Update the model configs in the config file.

    The below params can be the same or different from values in the pretrained model:
    - atomic_energy_shift
    - atomic_energy_scale
    - num_average_neigh

    All others should be exactly the same as values in the pretrained model.

    Args:
        config_ckpt: Config dict from checkpoint file.
        config: The entire config file.
        dataset: The dataset object.

    Returns:
        Update config dict.
    """
    # Params not in allowed should not be updated
    allowed = ["atomic_energy_shift", "atomic_energy_scale", "num_average_neigh"]
    for k in config["model"]:
        if k not in allowed:
            raise ValueError(
                f"When finetuning, `{k}` should not be provided, since it needs to be "
                f"the same as (and will be set to) the corresponding value in the "
                "pretrained model. Please remove it from the `model` section of the "
                "config file. The allowed keys in the `model` section of the config "
                f"file are: {allowed}."
            )

    # Copy params already provided in the `data` section of the config file
    ckpt_atomic_number = config["data"]["atomic_number"]
    r_cut = config["data"]["r_cut"]

    for k in ["atomic_number", "r_cut"]:
        if k in config["model"]:
            raise ValueError(
                f"Parameter {k} already provided in the `data` section of the "
                "config file. Please remove it from the `model` section."
            )

    # Atomic energy shift values
    # None: use values from the pretrained model
    # auto: use values determined from dataset
    # path to file: read atomic energy from file
    k = "atomic_energy_shift"
    shift_ckpt = config_ckpt["model"][k]
    shift = config["model"].pop(k, None)

    # Use values from pretrained model
    if shift is None:
        shift_final = shift_ckpt
    # Provide new values, either by linear fitting from dataset or reading from file
    elif isinstance(shift, str):
        if shift.lower() == "auto":
            dataset_shift = dataset.get_linear_fit_atomic_energy()
        else:
            # Read atomic energy from file; set to zero for atomic numbers not present
            if Path(shift).is_file():
                df = pd.read_json(shift)
                max_atomic_number = max(df["atomic_number"])
                dataset_shift = torch.zeros(max_atomic_number + 1)
                for _, row in df.iterrows():
                    dataset_shift[row["atomic_number"]] = row["energy"]
            else:
                raise ValueError(f"File does not exist: {shift}")

        # Set shift values
        # The size of shift should be equal to the number of supported atomic numbers
        # in the pretrained model.

        # For atomic number present in dataset, shift set to dataset value;
        # for atomic number not present, shift set to zero
        if shift_ckpt is None:
            shift_final = torch.zeros(max(ckpt_atomic_number) + 1)
            for i, v in enumerate(dataset_shift):
                shift_final[i] = v
            consistent = False

        # For atomic number present in dataset, shift set to dataset value;
        # for atomic number not present, shift set to value in the pretrained model
        else:
            dataset_atomic_number = dataset.get_atomic_number()

            consistent = True

            shift_final = copy.deepcopy(shift_ckpt)
            for i in dataset_atomic_number:
                if shift_final[i] != dataset_shift[i]:
                    shift_final[i] = dataset_shift[i]
                    consistent = False

        if not consistent:
            warnings.warn(
                f"Inconsistent `{k}`. Value in the pretrained is {shift_ckpt}, "
                f"whereas value from `{shift}` is: {dataset_shift}."
            )

    else:
        raise ValueError(f"`{k}` should be either 'null', 'auto', or path to a file.")

    # Atomic energy scale values
    # None: use values from the pretrained model
    # auto: use values determined from dataset
    k = "atomic_energy_scale"
    scale_ckpt = config_ckpt["model"][k]
    scale = config["model"].pop(k, None)

    # Use value from ckpt
    if scale is None:
        scale_final = scale_ckpt
    # Provide new value, obtained from dataset
    elif isinstance(scale, str):
        if scale.lower() == "auto":
            scale_final = dataset.get_root_mean_square_force()
        else:
            raise ValueError(f"`{k}` should be either `None` or `'auto'`; got {scale}")

        # If dataset value is different from pretrained model value, issue a warning.
        # This is legitimate, but we hope the user know it.
        if scale_ckpt is None:
            consistent = False
        else:
            if scale_final != scale_ckpt:
                consistent = False
            else:
                consistent = True
        if not consistent:
            warnings.warn(
                f"Inconsistent `{k}`. Value in the pretrained model is {scale_ckpt}, "
                f"whereas value from `{scale}` is {scale_final}."
            )
    else:
        raise ValueError(f"`{k}` should be either 'null' or 'auto'")

    # Number average neighbor
    # None: use value from the pretrained model
    # auto: use value determined from dataset
    k = "num_average_neigh"
    avg_neigh_ckpt = config_ckpt["model"][k]
    avg_neigh = config["model"].pop(k, None)

    if avg_neigh is None:
        avg_neigh_final = avg_neigh_ckpt
    elif isinstance(avg_neigh, str):
        if avg_neigh.lower() == "auto":
            avg_neigh_final = dataset.get_num_average_neigh()
        else:
            raise ValueError(f"`{k}` should be either 'null' or 'auto'")

        # If dataset value is different from ckpt value, issue an error.
        # This is legitimate, but we hope the user to know it.
        if avg_neigh_ckpt is None:
            consistent = False
        else:
            if avg_neigh_final != avg_neigh_ckpt:
                consistent = False
            else:
                consistent = True

        if not consistent:
            warnings.warn(
                f"Inconsistent `{k}`. "
                f"Value in the pretrained model is {avg_neigh_ckpt}, "
                f"whereas value from `{avg_neigh}` is {avg_neigh_final}."
            )

    else:
        raise ValueError(f"`{k}` should be either 'null' or 'auto'")

    # update config file
    config["model"]["num_atom_types"] = len(ckpt_atomic_number)
    config["model"]["r_cut"] = r_cut
    config["model"]["atomic_energy_shift"] = shift_final
    config["model"]["atomic_energy_scale"] = scale_final
    config["model"]["num_average_neigh"] = avg_neigh_final

    print(f"Updated model configs - `num_atom_types`: {len(ckpt_atomic_number)}")
    print(f"Updated model configs - `r_cut`: {r_cut}")
    print(f"Updated model configs - `atomic_energy_shift`: {shift_final}")
    print(f"Updated model configs - `atomic_energy_scale`: {scale_final}")
    print(f"Updated model configs - `num_average_neigh`: {avg_neigh_final}")

    return config


def check_configs(config: dict):
    if (
        config["loss"].get("stress_ratio", 0.0) > 1e-6
        and "stress" not in config["data"]["target_names"]
    ):
        raise ValueError(
            "Config inconsistence: `stress_ratio` is set in `loss`, but `stress` not "
            "provided in `data.target_names`."
        )


def main(config: dict):
    L.seed_everything(config["seed_everything"])

    # Set default dtype
    dtype = config.get("default_dtype", "float32")
    torch.set_default_dtype(getattr(torch, dtype))

    # Get config from pretrained model
    checkpoint = config.pop("restore_checkpoint", None)
    if checkpoint is None:
        raise ValueError("`restore_checkpoint` should be provided for fine-tuning.")
    else:
        d = torch.load(checkpoint, weights_only=True)
        config_ckpt = d["hyper_parameters"]["other_hparams"]

    # Load data
    config = update_data_configs(config_ckpt, config)
    with timer("data loading"):
        train_loader, val_loader, test_loader = get_dataloaders(**config["data"])

    # # Update model
    config = update_model_configs(config_ckpt, config, train_loader.dataset)

    # Add Additional info to the config for record-keeping
    config["git_commit"] = get_git_commit()

    # Consistence checking
    check_configs(config)

    # Create model
    # Override ALL hyperparameters from the config file
    names = {
        "loss": "loss_hparams",
        "metrics": "metrics_hparams",
        "ema": "ema_hparams",
        "optimizer": "optimizer_hparams",
        "lr_scheduler": "lr_scheduler_hparams",
    }
    overrides = {v: config.pop(k) for k, v in names.items()}
    overrides["other_hparams"] = config

    model = load_model(
        InteratomicPotentialLitModule,
        InteratomicPotential,
        checkpoint,
        overrides=overrides,
        params_load_mode="ema",
        reset_ema_step=True,
    )

    # Train
    try:
        callbacks = instantiate_class(config["trainer"].pop("callbacks"))
    except KeyError:
        callbacks = None

    try:
        logger = instantiate_class(config["trainer"].pop("logger"))

        ## For DEBUG only, should be commented out
        ## log gradients, parameter histogram and model topology
        ## For test run with small max_epoch, you might need to set `log_freq` to a
        ## smaller value (default is 100) so that this is executed at least once.
        # logger.watch(model, log="all", log_graph=False, log_freq=1)
    except KeyError:
        logger = None

    trainer = Trainer(callbacks=callbacks, logger=logger, **config["trainer"])

    # Train model
    with timer("model training"):
        trainer.fit(
            model,
            train_dataloaders=train_loader,
            val_dataloaders=val_loader,
            ckpt_path=None,
        )

    # Save the last epoch model
    # The behavior of  `save_last` in ModelCheckpoint callback is buggy; save manually
    trainer.save_checkpoint("./last_epoch.ckpt")

    # Test results on the best model determined by the validation set
    with timer("testing the best model"):
        out = trainer.test(ckpt_path="best", dataloaders=test_loader)
    print("Best model test results:", out)
    print(f"Best checkpoint path: {trainer.checkpoint_callback.best_model_path}")

    # Validation results on best model
    with timer("validating the best model"):
        out = trainer.validate(ckpt_path="best", dataloaders=val_loader)
    print("Best model val results:", out)


if __name__ == "__main__":

    config_file = Path(__file__).parent / "configs" / "config_ip_finetune.yaml"

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
