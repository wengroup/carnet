import torch
from natt.utils import is_symmetric_traceless, letter_index

from carnet.core.legendre import legendre
from carnet.core.unit_vector import Polyadics
from carnet.core.unit_vector_1 import (
    get_nt_from_vector,
    get_nt_from_vector_rule,
    get_polyadics_from_vector,
)
from carnet.core.unit_vector_2 import get_nt_from_vector as get_nt_from_vector_2
from carnet.core.unit_vector_2 import (
    get_polyadics_from_vector as get_polyadics_from_vector_2,
)


def test_get_nt_from_vector_rule():
    rule, symmetry, delta_indices = get_nt_from_vector_rule(3, 0)
    assert rule == "...a,...b,...c->...abc"
    assert symmetry == "xxx"
    assert delta_indices == ""

    rule, symmetry, delta_indices = get_nt_from_vector_rule(3, 1)
    assert rule == "...a,bc->...abc"
    assert symmetry == "xaa"
    assert delta_indices == "a"

    rule, symmetry, delta_indices = get_nt_from_vector_rule(4, 0)
    assert rule == "...a,...b,...c,...d->...abcd"
    assert symmetry == "xxxx"
    assert delta_indices == ""

    rule, symmetry, delta_indices = get_nt_from_vector_rule(4, 1)
    assert rule == "...a,...b,cd->...abcd"
    assert symmetry == "xxaa"
    assert delta_indices == "a"

    rule, symmetry, delta_indices = get_nt_from_vector_rule(4, 2)
    assert rule == "...,ab,cd->...abcd"
    assert symmetry == "aabb"
    assert delta_indices == "ab"


def test_get_nt_from_vector():
    a = torch.tensor([1.0, 2.0, 3.0])
    b = torch.tensor([3.2, 2.5, 1.5])
    a = a / a.norm()
    b = b / b.norm()
    a_dot_b = torch.dot(a, b)

    nt = get_nt_from_vector(a, 0)
    assert nt.shape == (1,)
    assert nt == torch.tensor([1.0])

    nt = get_nt_from_vector(a, 1)
    assert nt.shape == (3,)
    assert torch.allclose(nt, a)

    for n in range(2, 8):
        nt = get_nt_from_vector(a, n)
        assert nt.shape == (3,) * n
        assert is_symmetric_traceless(nt, atol=1e-5), f"Failing for n = {n}"

        # check n-contraction between nt and b is equal to Legendre(a\dot b)
        indices = letter_index(n)
        rule = "".join(indices) + "," + ",".join(indices)
        tp = torch.einsum(rule, nt, *([b] * n))
        leg = legendre(a_dot_b, n)
        assert torch.allclose(tp, leg), f"Failing for n = {n}"

        # check n-contraction between nt and a is equal to 1
        tp = torch.einsum(rule, nt, *([a] * n))
        assert torch.allclose(tp, torch.tensor(1.0)), f"Failing for n = {n}"


def test_get_nt_from_vector_batch():
    a = torch.tensor([1.0, 2.0, 3.0])
    a = a / a.norm()

    batch = 4
    a2 = a.repeat(batch, 1)

    for n in range(5):
        nt = get_nt_from_vector(a, n)
        if n == 0:
            assert nt.shape == (1,)
        else:
            assert nt.shape == (3,) * n

        nt_repeat = nt.repeat(batch, *([1] * n))
        bnt = get_nt_from_vector(a2, n)
        if n == 0:
            assert bnt.shape == (batch, 1)
        else:
            assert bnt.shape == (batch,) + (3,) * n

        assert torch.allclose(bnt, nt_repeat), f"Failing for n = {n}"


def test_get_polyadics_from_vector():
    a = torch.tensor([1.0, 2.0, 3.0])
    a = a / a.norm()

    for L in range(5):
        x = get_polyadics_from_vector(a, L)
        assert x.shape == ((3 ** (L + 1) - 1) // 2,)


def test_implementation():
    """
    Check different implementations of get_nt_from_vector produce the same result.
    """
    torch.manual_seed(35)

    a = torch.randn(2, 3)
    a /= a.norm(dim=-1, keepdim=True)

    for l in range(4):
        out1 = get_nt_from_vector(a, l)
        out2 = get_nt_from_vector_2(a, l)
        assert torch.allclose(out1, out2)


def test_polyadics():
    """
    Check Polyadics implementation produces the same result as the reference.
    """
    torch.manual_seed(35)
    a = torch.randn(10000, 3)
    a /= a.norm(dim=-1, keepdim=True)

    L = 4
    poly_mod = Polyadics(L, normalize="unity")
    out_poly = poly_mod(a)

    # Reference implementation
    out_ref_1 = get_polyadics_from_vector(a, L)

    out_ref_2 = get_polyadics_from_vector_2(a, L)

    assert torch.allclose(out_poly, out_ref_1, atol=1e-6)
    assert torch.allclose(out_poly, out_ref_2, atol=1e-6)
