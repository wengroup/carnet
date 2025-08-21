import numpy as np
import torch
from torch import Tensor


def get_voigt_projection_tensor() -> Tensor:
    """
    Get a projection tensor for converting a 3x3x3x3 elastic tensor to Voigt notation.

    Returns:
        A 3x3x3x3x6x6 tensor M such that,
        M_voigt[a,b] = M[i,j,k,l,a,b] * C[i,j,k,l] (sum over i,j,k,l)
    """
    # Voigt index mapping [i,j] -> a (0-based)
    voigt_map = {
        (0, 0): 0,
        (1, 1): 1,
        (2, 2): 2,
        (1, 2): 3,
        (2, 1): 3,
        (0, 2): 4,
        (2, 0): 4,
        (0, 1): 5,
        (1, 0): 5,
    }

    M = torch.zeros(3, 3, 3, 3, 6, 6)
    counts = torch.zeros((6, 6), dtype=torch.int32)

    # Build projection tensor
    for (i, j), a in voigt_map.items():
        for (k, l), b in voigt_map.items():
            M[i, j, k, l, a, b] += 1.0
            counts[a, b] += 1

    # Normalize by counts
    M = M / counts.reshape(1, 1, 1, 1, 6, 6)

    return M


if __name__ == "__main__":
    # Create a random elastic tensor
    from natt.sym import symmetrize

    torch.manual_seed(35)
    T = torch.randn(3, 3, 3, 3)
    T = symmetrize(T, "ijkl=jikl=klij")

    # get voigt tensor
    M = get_voigt_projection_tensor()
    T_v = torch.einsum("ijkl,ijklab->ab", T, M).numpy()

    # Get voigt tensor using pymatgen
    from pymatgen.analysis.elasticity.elastic import ElasticTensor

    ET = ElasticTensor(T)
    T_v_2 = ET.voigt

    assert np.allclose(T_v, T_v_2)
