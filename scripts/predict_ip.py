from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
import tqdm
from torch import Tensor
from torch_geometric.loader.dataloader import DataLoader

from carnet.data.dataset import DatasetIP
from carnet.data.transform import ConsecutiveAtomType
from carnet.model.ip import InteratomicPotential
from carnet.model.pl.pl_ip import InteratomicPotentialLitModule
from carnet.model.pl.utils import load_model


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


def get_references(filename: Path) -> tuple[Tensor, Tensor]:

    # Get references
    df = pd.read_json(filename)
    e_ref = df["energy"].to_list()
    f_ref = df["forces"].to_list()
    e_ref = torch.tensor(e_ref)
    f_ref = torch.cat([torch.tensor(x) for x in f_ref])

    n_atoms = torch.tensor(df["coords"].apply(len).to_list())

    return e_ref, f_ref, n_atoms


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

    # Device to run it
    device = "cpu"  # e.g. cpu, cuda

    # Get references and predictions
    e_ref, f_ref, n_atoms = get_references(filename)
    e_ref = e_ref.to(device)
    f_ref = f_ref.to(device)
    n_atoms = n_atoms.to(device)

    e_pred, f_pred = predict(filename, checkpoint, map_location=device, batch_size=20)

    # Overall metrics
    e_mae = torch.mean(torch.abs(e_ref - e_pred))
    e_mae_per_atom = torch.mean(torch.abs(e_ref - e_pred) / n_atoms)
    f_mae = torch.mean(torch.abs(f_ref - f_pred))
    print(f"MAE of energy: {e_mae:.4e}")
    print(f"MAE of energy/atom: {e_mae_per_atom:.4e}")
    print(f"MAE of forces: {f_mae:.4e}")

    e_rmse = torch.sqrt(torch.mean((e_ref - e_pred) ** 2))
    e_rmse_per_atom = torch.sqrt(torch.mean((e_ref / n_atoms - e_pred / n_atoms) ** 2))
    f_rmse = torch.sqrt(torch.mean((f_ref - f_pred) ** 2))
    print(f"RMSE of energy: {e_rmse:.4e}")
    print(f"RMSE of energy/atom: {e_rmse_per_atom:.4e}")
    print(f"RMSE of forces: {f_rmse:.4e}")

    # Distribution of energy errors
    # e_diff = e_pred - e_ref
    # plot_hist(e_diff.detach().numpy(), "E_pred - E_ref", "energy_diff")
