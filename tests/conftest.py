import pytest
import torch

from carten.core.reduce import symmetrize_and_remove_trace


@pytest.fixture(scope="session")
def T0():
    return torch.tensor(1.0)


@pytest.fixture(scope="session")
def T1():
    return get_T(1)


@pytest.fixture(scope="session")
def T2():
    return get_T(2)


@pytest.fixture(scope="session")
def T3():
    return get_T(3)


@pytest.fixture(scope="session")
def T4():
    return get_T(4)


@pytest.fixture(scope="session")
def NT0():
    return torch.tensor(1.0)


@pytest.fixture(scope="session")
def NT1():
    return get_NT(1)


@pytest.fixture(scope="session")
def NT2():
    return get_NT(2)


@pytest.fixture(scope="session")
def NT3():
    return get_NT(3)


@pytest.fixture(scope="session")
def NT4(T4):
    return get_NT(4)


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


def get_T(rank: int):
    """Create a tensor of rank `rank` for testing."""
    t = torch.arange(3**rank).reshape([3] * rank).to(torch.float32)
    return t / t.mean()


def get_NT(rank: int):
    """Create a natural tensor of rank `rank` for testing."""
    t = get_T(rank)
    return symmetrize_and_remove_trace(t)
