from pathlib import Path

import numpy as np
import pandas as pd
import torch
import tqdm
from torch import Tensor
from torch_geometric.loader.dataloader import DataLoader

from carten.data.dataset import Dataset
from carten.data.transform import ConsecutiveAtomType
from carten.model.pl.pl_tensor_model import StructureTensorLitModule
from carten.model.pl.utils import load_model
from carten.model.tensor_model import StructureTensorModel


def get_dataset(
    filename: Path, target_name: str, atomic_number: list[int], r_cut: float
):
    dataset = Dataset(
        filename=filename,
        target_names=(target_name,),
        r_cut=r_cut,
        transform=ConsecutiveAtomType(atomic_number),
        log=False,
    )

    return dataset


def get_dataloader(
    filename: Path,
    target_name: str,
    atomic_number: list[int],
    r_cut: float,
    batch_size: int,
):
    dataset = get_dataset(filename, target_name, atomic_number, r_cut)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    return loader


def predict(
    target_name,
    filename: Path,
    checkpoint: Path,
    map_location: str = "cpu",
    batch_size: int = 10,
):
    """
    Args:
        target_name: Name of the target to predict.
        filename: Path to the file containing the structures to make predictions.
        checkpoint: Path to the checkpoint file.
        map_location: Device to load the model on and run the predictions.
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
        StructureTensorLitModule, StructureTensorModel, checkpoint, map_location
    )

    output = []
    for batch in tqdm.tqdm(loader):
        batch = batch.to(model.device)
        out = model(batch)
        output.append(out)

    output = process_dict(output)

    return output


def compute_metrics(target_name, filename, checkpoint):
    """Compute the MAEs of energy and forces."""

    # Get references
    df = pd.read_json(filename)
    ref = df[target_name].to_list()
    ref = process_dict(ref)

    # Get predictions
    pred = predict(target_name, filename, checkpoint)

    for k, v_r in ref.items():
        v_p = pred[int(k)].detach().numpy()
        mae = np.mean(np.abs(v_r - v_p))
        print(f"MAE of {k}: {mae:.4f}")


def process_dict(data: list[dict[int, np.ndarray]]):
    """
    Process list of dict of array to get dict of array, where the out-most list is
    concated.
    """
    out = {}
    for d in data:
        for k, v in d.items():
            if k not in out:
                out[k] = []
            out[k].append(v)

    for k, v in out.items():
        if isinstance(v[0], np.ndarray):
            out[k] = np.concatenate(v, axis=0)
        elif isinstance(v[0], Tensor):
            out[k] = torch.cat(v, dim=0)

    return out


if __name__ == "__main__":

    target_name = "elastic_tensor_natural"

    filename = "/Users/mjwen/Packages/carten_analysis/dataset/elastic_tensor/20230504/crystal_elasticity_filtered_n20.json"

    # To generate an example checkpoint, run `train_ip.py` first and then checkout
    # `./carten_proj` to the checkpoint you want to use.
    checkpoint = "./carten_proj/1m1g71f8/checkpoints/epoch=1-step=6.ckpt"

    compute_metrics(target_name, filename, checkpoint)
