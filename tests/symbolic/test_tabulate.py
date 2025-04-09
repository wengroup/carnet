import numpy as np
import pytest

from carten.symbolic.tabulate import get_G_H_S


# TODO, n=1 does not work
@pytest.mark.parametrize("rank", [2, 3, 4])
def test_get_G_H_S(rank):
    """Test the get_G_H_S function.

    For a given tensor T, obtain the natural tensors X (using H), and then obtain the
    embedding T' in the original tensor space (using G). We check that we can recover T.

    Args:
        rank: rank of the tensor T
    """
    np.random.seed(35)

    T = np.random.randn(*[3] * rank)

    output = get_G_H_S(rank)

    all_T_prime = []
    for j, out_j in output.items():

        for p, (H, G, S) in enumerate(zip(out_j["H"], out_j["G"], out_j["S"])):
            # X = H T
            X = np.einsum(H["rule"], H["numerical"], T)

            # T' = G X
            T_p_1 = np.einsum(G["rule"], G["numerical"], X)

            # T' = S T
            T_p_2 = np.einsum(S["rule"], S["numerical"], T)

            # T_p_1 and T_p_2 should be equal
            assert np.allclose(T_p_1, T_p_2), (
                f"T_p_1 and T_p_2 are not equal for j=" f"{j}, p={p}"
            )

            all_T_prime.append(T_p_1)

    sum_T_prime = np.sum(all_T_prime, axis=0)

    assert np.allclose(sum_T_prime, T)
