from math import factorial as factorial_math

import torch

from carten.utils import double_index, factorial, double_factorial, get_trace


def test_multi_double_index():
    assert double_index(2) == ["ab", "cd"]
    assert double_index(3, start=1) == ["bc", "de", "fg"]


def test_factorial():
    for i in range(10):
        assert factorial(i) == factorial_math(i)


def test_double_factorial():
    assert double_factorial(0) == 1
    assert double_factorial(1) == 1
    assert double_factorial(2) == 2
    assert double_factorial(3) == 3
    assert double_factorial(4) == 8
    assert double_factorial(5) == 15
    assert double_factorial(6) == 48
    assert double_factorial(7) == 105
    assert double_factorial(8) == 384

    assert double_factorial(7, lower_bound=3) == 105
    assert double_factorial(8, lower_bound=4) == 192

    for i in range(5, 10):
        assert double_factorial(i) // double_factorial(i - 4) == double_factorial(
            i, lower_bound=i - 4 + 2
        )


def test_get_trace(T2, T3):
    trace = get_trace(T2, i=0, j=1)
    assert torch.allclose(trace, torch.tensor([12.0]))

    trace = get_trace(T3, i=0, j=1)
    assert torch.allclose(trace, torch.tensor([36.0, 39.0, 42.0]))

    trace = get_trace(T3, i=1, j=2)
    assert torch.allclose(trace, torch.tensor([12.0, 39.0, 66.0]))
