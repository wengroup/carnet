import pytest
import torch

from carten.reduce import symmetrize_and_remove_trace


@pytest.fixture(scope="session")
def T0():
    return torch.tensor(1.0)


@pytest.fixture(scope="session")
def T1():
    return torch.arange(3).to(torch.float32)


@pytest.fixture(scope="session")
def T2():
    return torch.arange(9).reshape((3, 3)).to(torch.float32)


@pytest.fixture(scope="session")
def T3():
    return torch.arange(27).reshape((3, 3, 3)).to(torch.float32)


@pytest.fixture(scope="session")
def T4():
    return torch.arange(81).reshape((3, 3, 3, 3)).to(torch.float32)


@pytest.fixture(scope="session")
def NT0(T0):
    return T0


@pytest.fixture(scope="session")
def NT1(T1):
    return symmetrize_and_remove_trace(T1)


@pytest.fixture(scope="session")
def NT2(T2):
    return symmetrize_and_remove_trace(T2)


@pytest.fixture(scope="session")
def NT3(T3):
    return symmetrize_and_remove_trace(T3)


@pytest.fixture(scope="session")
def NT4(T4):
    return symmetrize_and_remove_trace(T4)


@pytest.fixture(scope="session")
def T0_chunk(NT0, mul=2):
    # adding a dummy leading dimension 1
    return torch.stack([NT0, NT0]).reshape(1, mul * 1)


@pytest.fixture(scope="session")
def T0_shaped_chunk(NT0, mul=2):
    # adding a dummy leading dimension 1
    return torch.stack([NT0, NT0]).reshape(1, mul, 1)


@pytest.fixture(scope="session")
def T1_chunk(NT1, mul=2):
    return torch.cat([NT1.flatten()] * mul, dim=-1).reshape(1, mul * 3)


@pytest.fixture(scope="session")
def T1_shaped_chunk(NT1, mul=2):
    return torch.cat([NT1.flatten()] * mul, dim=-1).reshape(1, mul, 3)


@pytest.fixture(scope="session")
def T2_chunk(NT2, mul=2):
    return torch.cat([NT2.flatten()] * mul, dim=-1).reshape(1, mul * 9)


@pytest.fixture(scope="session")
def T2_shaped_chunk(NT2, mul=2):
    return torch.cat([NT2.flatten()] * mul, dim=-1).reshape(1, mul, 3, 3)


@pytest.fixture(scope="session")
def T3_chunk(NT3, mul=2):
    return torch.cat([NT3.flatten()] * mul, dim=-1).reshape(1, mul * 27)


@pytest.fixture(scope="session")
def T3_shaped_chunk(NT3, mul=2):
    return torch.cat([NT3.flatten()] * mul, dim=-1).reshape(1, mul, 3, 3, 3)


@pytest.fixture(scope="session")
def T4_chunk(NT4, mul=2):
    return torch.cat([NT4.flatten()] * mul, dim=-1).reshape(1, mul * 81)


@pytest.fixture(scope="session")
def T4_shaped_chunk(T4, mul=2):
    return torch.cat([T4.flatten()] * mul, dim=-1).reshape(1, mul, 3, 3, 3, 3)
