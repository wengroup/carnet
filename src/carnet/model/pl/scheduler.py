import torch
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, LinearLR, SequentialLR


def LinearWarmupCosineAnnealingWarmRestarts(
    optimizer, start_factor: float, warmup_epochs: int, **kwargs
):
    """
    Create a learning rate scheduler that combines linear warmup and cosine annealing
    with warm restarts.

    Args:
        kwargs: keyword arguments for the CosineAnnealingWarmRestarts scheduler.
    """
    return SequentialLR(
        optimizer,
        [
            LinearLR(
                optimizer, start_factor, end_factor=1.0, total_iters=warmup_epochs
            ),
            CosineAnnealingWarmRestarts(optimizer, **kwargs),
        ],
        [warmup_epochs],
    )


if __name__ == "__main__":

    optimizer = torch.optim.SGD([torch.nn.Parameter(torch.tensor(0.0))], lr=0.01)

    scheduler = LinearWarmupCosineAnnealingWarmRestarts(
        optimizer=optimizer,
        start_factor=0.01,
        warmup_epochs=10,
        T_0=100,
        T_mult=1,
        eta_min=0.0,
        last_epoch=-1,
    )

    # Generate a matplotlib plot to visualize the learning rate schedule
    import matplotlib.pyplot as plt
    import numpy as np

    epochs = np.arange(0, 210)

    lrs = []
    for i in epochs:
        lrs.append(scheduler.get_last_lr()[0])
        optimizer.step()
        scheduler.step()

    # create a pdf file
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, lrs, "o", label="Learning Rate")
    plt.title("Learning Rate Schedule")
    plt.xlabel("Epochs")
    plt.ylabel("Learning Rate")
    plt.savefig("learning_rate_schedule.pdf")
