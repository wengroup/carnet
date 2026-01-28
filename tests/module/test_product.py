import torch

from carnet.module.product import TensorProduct
from carnet.module.product1 import TensorProduct as TensorProduct1
from carnet.module.product2 import TensorProduct as TensorProduct2


def test_tensor_product_standard(NT0, NT1, NT2, NT3):
    """Test standard TensorProduct (for_atomic_moment=False)."""
    B = 3  # batch size
    F = 2  # feature size

    NT0 = NT0.view(-1)
    NT1 = NT1.view(-1)
    NT2 = NT2.view(-1)
    NT3 = NT3.view(-1)

    # Input x: (B, F, T)
    x = torch.cat([NT0, NT1, NT2, NT3], dim=-1).repeat(B, F, 1)

    L1 = L2 = L3 = 3
    tp = TensorProduct(F, L1, L2, L3, for_atomic_moment=False)
    tp1 = TensorProduct1(F, L1, L2, L3, for_atomic_moment=False)
    tp2 = TensorProduct2(F, L1, L2, L3, for_atomic_moment=False)

    # Synchronize kernel weights
    for k, k1, k2 in zip(tp.kernels, tp1.kernels, tp2.kernels):
        if k is not None:
            k1.weight.data.copy_(k.weight.data)
            k2.weight.data.copy_(k.weight.data)

    # Case: R not used (None)
    z = tp(x, x)
    z1 = tp1(x, x)
    z2 = tp2(x, x)

    # Check consistency between implementations
    assert torch.allclose(z, z1, atol=1e-6)
    assert torch.allclose(z, z2, atol=1e-6)

    # Batching dimension should not affect the output since input is repeated
    assert torch.allclose(z[0], z[1])
    assert torch.allclose(z[0], z[2])


def test_tensor_product_atomic_moment(NT0, NT1, NT2, NT3):
    """Test TensorProduct for atomic moments (for_atomic_moment=True)."""
    B = 3  # batch size
    F = 2  # feature size

    NT0 = NT0.view(-1)
    NT1 = NT1.view(-1)
    NT2 = NT2.view(-1)
    NT3 = NT3.view(-1)

    # Input x: (B, F, T)
    x = torch.cat([NT0, NT1, NT2, NT3], dim=-1).repeat(B, F, 1)
    # Input y: (B, T) for atomic moment
    y = torch.cat([NT0, NT1, NT2, NT3], dim=-1).repeat(B, 1)

    L1 = L2 = L3 = 3
    tp = TensorProduct(F, L1, L2, L3, for_atomic_moment=True)
    tp1 = TensorProduct1(F, L1, L2, L3, for_atomic_moment=True)
    tp2 = TensorProduct2(F, L1, L2, L3, for_atomic_moment=True)

    # Synchronize kernel weights
    for k, k1, k2 in zip(tp.kernels, tp1.kernels, tp2.kernels):
        if k is not None:
            k1.weight.data.copy_(k.weight.data)
            k2.weight.data.copy_(k.weight.data)

    # Weights R for atomic moment (required)
    all_paths = []
    for l3 in tp.L3:
        all_paths.extend(tp.paths[l3])

    num_paths = len(all_paths)
    # Same weights across batch for batch invariance test
    R_tensor = torch.randn(1, num_paths, F).repeat(B, 1, 1)

    z = tp(x, y, R=R_tensor)
    z1 = tp1(x, y, R=R_tensor)
    z2 = tp2(x, y, R=R_tensor)

    # Check consistency between implementations
    assert torch.allclose(z, z1, atol=1e-6)
    assert torch.allclose(z, z2, atol=1e-6)

    # Batching dimension should not affect the output since input is repeated
    assert torch.allclose(z[0], z[1])
    assert torch.allclose(z[0], z[2])
