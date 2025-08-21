import torch

from carnet.module.product import TensorProduct


def test_TensorProduct(NT0, NT1, NT2, NT3):
    B = 3  # batch size
    F = 2  # feature size

    NT0 = NT0.view(-1)
    NT1 = NT1.view(-1)
    NT2 = NT2.view(-1)
    NT3 = NT3.view(-1)

    x = torch.cat([NT0, NT1, NT2, NT3], dim=-1).repeat(B, F, 1)

    tp = TensorProduct(F, 3, 3, 3)
    z = tp(x, x)  # (B, F, T3)
    assert z.shape == (B, F, (3**4 - 1) // 2)

    # Batching dimension should not affect the output
    assert torch.allclose(z[0], z[1])
    assert torch.allclose(z[0], z[2])
