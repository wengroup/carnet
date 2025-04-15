from pathlib import Path

import numpy as np
import pandas as pd
import torch
import tqdm
from torch_geometric.loader.dataloader import DataLoader

from carten.data.dataset import Dataset
from carten.data.transform import ConsecutiveAtomType
from carten.model.ip import InteratomicPotential
from carten.model.pl.pl_ip import InteratomicPotentialLitModule
from carten.model.pl.utils import load_model


def get_dataset(filename: Path, atomic_number: list[int], r_cut: float):
    dataset = Dataset(
        filename=filename,
        target_names=("energy", "forces"),
        r_cut=r_cut,
        transform=ConsecutiveAtomType(atomic_number),
        log=False,
    )

    return dataset


def get_dataloader(
    filename: Path, atomic_number: list[int], r_cut: float, batch_size: int
):
    dataset = get_dataset(filename, atomic_number, r_cut)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    return loader


def predict(
    filename: Path, checkpoint: Path, map_location: str = "cpu", batch_size: int = 10
):
    """
    Args:
        filename: Path to the file containing the structures to make predictions.
        checkpoint: Path to the checkpoint file.
        map_location: Device to load the model on and run the predictions.
    """

    d = torch.load(checkpoint, map_location=map_location, weights_only=True)
    data_config = d["hyper_parameters"]["other_hparams"]["data"]

    loader = get_dataloader(
        filename=filename,
        atomic_number=data_config["atomic_number"],
        r_cut=data_config["r_cut"],
        batch_size=batch_size,
    )

    model = load_model(
        InteratomicPotentialLitModule, InteratomicPotential, checkpoint, map_location
    )

    energy = []
    forces = []
    for batch in tqdm.tqdm(loader):
        batch.pos.requires_grad_(True)

        batch = batch.to(model.device)
        e_pred, f_pred = model(batch)

        energy.extend(e_pred.cpu().detach().numpy())
        forces.extend(f_pred.cpu().detach().numpy())

    return energy, forces


def compute_metrics(filename, checkpoint):
    """Compute the MAEs of energy and forces."""

    # Get references
    df = pd.read_json(filename)
    e_ref = df["energy"].to_numpy()
    f_ref = df["forces"].to_list()
    f_ref = np.concatenate(f_ref, axis=0)

    # Get predictions
    e_pred, f_pred = predict(filename, checkpoint)

    e_mae = np.mean(np.abs(e_ref - e_pred))
    f_mae = np.mean(np.abs(f_ref - f_pred))

    print(f"MAE of energy: {e_mae:.4f} eV")
    print(f"MAE of forces: {f_mae:.4f} eV/Å")


if __name__ == "__main__":

    filename = "/Users/mjwen/Packages/camp_analysis/dataset/nequip_LiPS/json_data/train_100_LiPS.json"

    # To generate an example checkpoint, run `train_ip.py` first and then checkout
    # `./carten_proj` to the checkpoint you want to use.
    checkpoint = "./carten_proj/0uvr7vze/checkpoints/epoch=1-step=6.ckpt"

    compute_metrics(filename, checkpoint)
