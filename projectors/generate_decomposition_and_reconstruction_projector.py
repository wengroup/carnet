from pathlib import Path

import torch
from natt.GHS import get_G_H_S

from carnet.utils import yaml_dump

torch.set_default_dtype(torch.float64)


def convert_to_list(tensor):
    if isinstance(tensor, torch.Tensor):
        return tensor.tolist()
    elif isinstance(tensor, dict):
        if "symbolic" in tensor:
            for k, v in tensor.items():
                if k == "symbolic":
                    tensor[k] = str(v).replace("\u03b4", "d").replace("\u03b5", "e")
                else:
                    tensor[k] = convert_to_list(v)
            return tensor
        else:
            return {k: convert_to_list(v) for k, v in tensor.items()}
    elif isinstance(tensor, list):
        return [convert_to_list(v) for v in tensor]
    else:
        return tensor


if __name__ == "__main__":
    physical_tensors = {
        "nuclear_shielding": {
            "rank": 2,
            "symmetry": "ij",
        },
        "polarizability": {
            "rank": 2,
            "symmetry": "ij=ji",
        },
        "piezoelectricity": {
            "rank": 3,
            "symmetry": "ijk=ikj",
        },
        "elasticity": {
            "rank": 4,
            "symmetry": "ijkl=jikl=klij",
        },
    }

    results = {}
    for name, info in physical_tensors.items():
        out = get_G_H_S(info["rank"], info["symmetry"], numerical=True)
        info["GHS"] = convert_to_list(out)
        results[name] = info

    # Same to yaml
    filename = Path("./decomposition_and_reconstruction_projector.yaml")
    yaml_dump(results, filename)
