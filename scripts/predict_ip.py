from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
import tqdm
from torch import Tensor
from torch_geometric.loader.dataloader import DataLoader

from carten.data.dataset import DatasetIP
from carten.data.transform import ConsecutiveAtomType
from carten.model.ip import InteratomicPotential
from carten.model.pl.pl_ip import InteratomicPotentialLitModule
from carten.model.pl.utils import load_model


def get_dataloader(
    filename: Path, atomic_number: list[int], r_cut: float, batch_size: int
):
    """
    Get the dataset and loader for prediction.

    The dataset should be provided in a file of the same format as the train/val/test
    set.
    """

    dataset = DatasetIP(
        filename=filename,
        target_names=["energy", "forces"],
        r_cut=r_cut,
        transform=ConsecutiveAtomType(atomic_number),
        log=False,
    )

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    return loader


def predict(
    filename: Path, checkpoint: Path, map_location: str = "cpu", batch_size: int = 20
) -> tuple[Tensor, Tensor]:
    """
    Predict energy and forces.

    Args:
        filename: Path to the file containing the structures to make predictions.
        checkpoint: Path to the checkpoint file.
        map_location: Device to load the model on and run the predictions.

    Returns:
        energy: shape (N,), energy of each configuration
        forces: shape (Na, 3), forces on atoms in all N configurations
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
        e_pred, f_pred = model.forward_ema(batch)

        energy.append(e_pred.detach())
        forces.append(f_pred.detach())

    return torch.cat(energy), torch.cat(forces)


def compute_metrics(filename: Path, checkpoint: Path):
    """Compute the MAEs of energy and forces.
    Args:
        filename: Path to the file containing the dataset to make predictions.
        checkpoint: Path to the checkpoint file.
    """

    # Get references
    df = pd.read_json(filename)
    e_ref = df["energy"].to_list()
    f_ref = df["forces"].to_list()
    e_ref = torch.tensor(e_ref)
    f_ref = torch.cat([torch.tensor(x) for x in f_ref])

    # Get predictions
    e_pred, f_pred = predict(filename, checkpoint)

    # Overall metrics
    e_mae = torch.mean(torch.abs(e_ref - e_pred))
    f_mae = torch.mean(torch.abs(f_ref - f_pred))
    print(f"MAE of energy: {e_mae:.4f}")
    print(f"MAE of forces: {f_mae:.4f}")

    # Distribution of energy errors
    e_diff = e_pred - e_ref
    plot_hist(e_diff.detach().numpy(), "E_pred - E_ref", "energy_diff")


def plot_hist(data, x_label, title: str, filename=None):
    """Create a histogram of the data and save it to a file."""

    fig, ax = plt.subplots()
    ax.hist(data, bins=100)

    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel("Counts")

    if filename is None:
        filename = f"{title}.pdf"

    fig.savefig(filename, bbox_inches="tight")


if __name__ == "__main__":

    filename = (
        "/Users/mjwen/Packages/camp_analysis/dataset/nequip_LiPS/train_100_LiPS.json"
    )

    # To generate an example checkpoint, run `train_ip.py` first
    checkpoint = "./last_epoch.ckpt"

    compute_metrics(filename, checkpoint)
