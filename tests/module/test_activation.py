import math

import matplotlib.pylab as plt
import torch

from carnet.module.activation import elu, relu, shifted_softplus, silu


def assert_one(func, x: float, target):
    x = torch.tensor(x).reshape(1, 1)
    y = func(x)
    assert torch.allclose(y[0, 0], torch.tensor(target))


def _plot(func, xmin=-3, xmax=3):

    # get data
    x = torch.linspace(xmin, xmax, 100).reshape(-1, 1)
    y = func(x)
    x = x.numpy().flatten()
    y = y.numpy().flatten()

    # plot
    fig, ax = plt.subplots()
    ax.plot(x, y)
    ax.grid(True)

    fig.savefig(f"{func.__name__}.pdf")


def test_elu():
    assert_one(elu, -1.0, math.exp(-1.0) - 1)
    assert_one(elu, 0.0, 0.0)
    assert_one(elu, 1.0, 1.0)

    _plot(elu)


def test_relu():
    assert_one(relu, -1.0, 0.0)
    assert_one(relu, 0.0, 0.0)
    assert_one(relu, 1.0, 1.0)

    _plot(relu)


def test_silu():
    f = lambda x: x / (1 + math.exp(-x))

    assert_one(silu, -1.0, f(-1.0))
    assert_one(silu, 0.0, 0.0)
    assert_one(silu, 1.0, f(1.0))

    _plot(silu)


def test_shifted_softplus():
    f = lambda x: math.log((1 + math.exp(x)) / 2)

    assert_one(shifted_softplus, -1.0, f(-1.0))
    assert_one(shifted_softplus, 0.0, 0.0)
    assert_one(shifted_softplus, 1.0, f(1.0))

    _plot(shifted_softplus)
