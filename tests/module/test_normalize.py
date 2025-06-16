import torch

from carten.module.normalize import LayerNorm


def test_normalize():
    # Create a LayerNorm instance
    F = 4
    layer_norm = LayerNorm(dim=F, slice_sizes=[1, 3, 9], eps=0)

    # Create a sample input tensor
    torch.manual_seed(35)
    data = torch.arange(F * (1 + 3 + 9), dtype=torch.float32).reshape(F, 1 + 3 + 9)
    input_tensor = torch.stack([data, data], dim=0)  # Shape: (batch_size, F, T)

    # Forward pass
    output_tensor = layer_norm(input_tensor)

    # Check the shape of the output tensor
    assert output_tensor.shape == input_tensor.shape, "Output shape mismatch"

    # Check scalars are corrected normalized, subtracting mean and dividing by std
    scalars = data[:, 0]
    mean = torch.mean(scalars)
    std = torch.sqrt(torch.mean((scalars - mean) ** 2))

    normalized = (scalars - mean) / std
    ref = torch.stack([normalized, normalized], dim=0)

    pred = output_tensor[..., 0]

    assert torch.allclose(ref, pred, atol=1e-6)
