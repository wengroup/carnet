from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
import tqdm
from torch import Tensor
from torch_geometric.loader.dataloader import DataLoader

from carnet.core.convert import Converter
from carnet.data.dataset import Dataset
from carnet.data.transform import ConsecutiveAtomType
from carnet.model.pl.pl_tensor_model import AtomicTensorLitModule
from carnet.model.pl.utils import load_model
from carnet.model.tensor_model import AtomicTensorModel


def get_dataloader(
    filename: Path,
    target_name: str,
    atomic_number: list[int],
    r_cut: float,
    batch_size: int,
):
    """
    Get the dataset and loader for prediction.

    The dataset should be provided in a file of the same format as the train/val/test
    set.
    """

    dataset = Dataset(
        filename=filename,
        target_names=(target_name, "atomic_selector"),
        r_cut=r_cut,
        transform=ConsecutiveAtomType(atomic_number),
        log=False,
    )

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    return loader


def predict(
    target_name,
    filename: Path,
    checkpoint: Path,
    map_location: str = "cpu",
    batch_size: int = 10,
) -> dict[int, Tensor]:
    """
    Predict the target using the model.

    Args:
        target_name: Name of the target to predict.
        filename: Path to the file containing the dataset to make predictions.
        checkpoint: Path to the checkpoint file.
        map_location: Device to load the model on and run the predictions.

    Returns:
        {rank: value}, where rank is the rank of the tensor and value is the predicted
        natural tensor corresponding to the rank. Each tensor value is of the shape
        (N, M, dim), where N is the number of data points, M is seniority of the natural
        tensors, and dim = 3^rank is the flattened dimension of the tensor.
    """

    d = torch.load(checkpoint, map_location=map_location, weights_only=True)
    data_config = d["hyper_parameters"]["other_hparams"]["data"]

    loader = get_dataloader(
        filename=filename,
        target_name=target_name,
        atomic_number=data_config["atomic_number"],
        r_cut=data_config["r_cut"],
        batch_size=batch_size,
    )

    model = load_model(
        AtomicTensorLitModule, AtomicTensorModel, checkpoint, map_location
    )

    output = []
    for batch in tqdm.tqdm(loader):
        batch = batch.to(model.device)
        out = model.forward_ema(batch)
        # detach to save memory
        out = {k: v.detach() for k, v in out.items()}
        output.append(out)

    output = process_dict(output)

    return output


def compute_metrics(target_name, symmetry, filename, checkpoint):
    """Compute the metrics using the predictions and references.

    Args:
        target_name: Name of the target to predict.
        symmetry: Symmetry of the target tensor. e.g. None for NMR tensor, meaning
            there is no symmetry.
        filename: Path to the file containing the dataset to make predictions.
        checkpoint: Path to the checkpoint file.
    """

    # Get references
    df = pd.read_json(filename)
    ref = df[target_name].to_list()
    ref = process_dict(ref)
    ref = {int(k): v for k, v in ref.items()}

    # Get predictions
    pred = predict(target_name, filename, checkpoint)

    # Metrics on natural tensors
    for k in ref:
        v_r = ref[k]
        v_p = pred[k].detach()

        # Overall MAE in the natural tensor space
        mae = torch.mean(torch.abs(v_r - v_p))
        print(f"Natural tensor MAE (rank={k}): {mae:.4f}")

        # Distribution of MAE of each data point, taking the average over tensor dim
        # and seniority dim, but not over data point dim
        mae = torch.mean(torch.abs(v_r - v_p), axis=tuple(range(1, v_r.ndim)))
        plot_hist(
            mae.detach().numpy(), "MAE of each structure", f"natural_MAE_rank" f"={k}"
        )

    # Metrics on ordinary tensors
    converter = Converter(symmetry)
    ref = converter.to_ordinary_tensor(ref)
    pred = converter.to_ordinary_tensor(pred)

    mae = torch.mean(torch.abs(ref - pred))
    print(f"Ordinary tensor MAE: {mae:.4f}")

    # Distribution of MAE of each data point
    mae = torch.mean(torch.abs(ref - pred), axis=tuple(range(1, ref.ndim)))
    plot_hist(mae.detach().numpy(), "MAE of each structure", "ordinary_MAE")


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


def process_dict(data: list[dict[int, Tensor]]) -> dict[int, Tensor]:
    """
    Process a list of dict of array to get a dict of array, where the out-most list is
    concatenated.
    """
    out = {}
    for d in data:
        for k, v in d.items():
            if k not in out:
                out[k] = []
            if not isinstance(v, Tensor):
                v = torch.tensor(v)
            out[k].append(v)

    for k, v in out.items():
        out[k] = torch.cat(v, dim=0)

    return out


if __name__ == "__main__":

    target_name = "nmr_tensor_natural"

    filename = "/Users/mjwen/Packages/carnet_analysis/dataset/nmr_tensor/20250424/nmr_tensor_n20.json"

    # To generate an example checkpoint, first run `train_atomic_tensor.py` and then
    # checkout `./carnet_proj` to get the checkpoint you want to use.
    checkpoint = "./carnet_proj/n228kouh/checkpoints/epoch=1-step=10.ckpt"

    symmetry = "ij"  # for NMR tensor
    compute_metrics(target_name, symmetry, filename, checkpoint)
