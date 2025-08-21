import torch
from natt.utils import is_symmetric, is_traceless, letter_index

from carnet.core.legendre import legendre
from carnet.core.unit_vector import get_nt_from_vector
from carnet.legacy.tensor_product_unbatched import (
    get_tp_even_rule,
    get_tp_odd_rule,
    tp_even,
    tp_odd,
)


def test_tp_rule_even():
    rule, symmetry, _ = get_tp_even_rule(4, 4, 4, 0)
    assert rule == "abcd,abcd->"
    assert symmetry == ""

    rule, symmetry, _ = get_tp_even_rule(4, 4, 3, 0)
    assert rule == "abcd,abce->de"
    assert symmetry == "xy"

    rule, symmetry, _ = get_tp_even_rule(4, 4, 3, 1)
    assert rule == "abcd,abcd,ef->ef"
    assert symmetry == "aa"

    rule, symmetry, _ = get_tp_even_rule(4, 4, 2, 0)
    assert rule == "abcd,abef->cdef"
    assert symmetry == "xxyy"

    rule, symmetry, _ = get_tp_even_rule(4, 4, 2, 1)
    assert rule == "abcd,abce,fg->defg"
    assert symmetry == "xyaa"

    rule, symmetry, _ = get_tp_even_rule(4, 4, 2, 2)
    assert rule == "abcd,abcd,ef,gh->efgh"
    assert symmetry == "aabb"


def test_tp_rule_odd():
    rule, symmetry, _ = get_tp_odd_rule(4, 4, 3, 0)
    assert rule == "abc,bdef,cdef->a"
    assert symmetry == "x"

    rule, symmetry, _ = get_tp_odd_rule(4, 4, 2, 0)
    assert rule == "abc,bdef,cdeg->afg"
    assert symmetry == "xyz"

    rule, symmetry, _ = get_tp_odd_rule(4, 4, 2, 1)
    assert rule == "abc,bdef,cdef,gh->agh"
    assert symmetry == "xaa"


def test_tp_even(NT3, NT4):
    for i in [1, 3, 5, 7]:
        out = tp_even(NT3, NT4, out_rank=i, normalize="unity")
        assert out.ndim == i
        assert is_symmetric(out, atol=1e-6), f"Not symmetric for {i}"
        assert is_traceless(out, atol=1e-5), f"Not traceless for {i}"

    # for i in [0, 2, 4, 6, 8]:
    for i in [0, 2, 4, 6]:
        out = tp_even(NT4, NT4, out_rank=i, normalize="unity")
        assert out.ndim == i
        assert is_symmetric(out, atol=1e-6), f"Not symmetric for {i}"
        assert is_traceless(out, atol=1e-5), f"Not traceless for {i}"


def test_tp_odd(NT3, NT4):
    for i in [2, 4, 6]:
        out = tp_odd(NT3, NT4, out_rank=i, normalize="unity")
        assert out.ndim == i
        assert is_symmetric(out, atol=1e-6), f"Failed symmetry for {i}"
        assert is_traceless(out, atol=1e-5), f"Failed traceless for {i}"

    for i in [1, 3, 5, 7]:
        out = tp_odd(NT4, NT4, out_rank=i, normalize="unity")
        assert out.ndim == i
        assert is_symmetric(out, atol=1e-6), f"Failed symmetry for {i}"
        assert is_traceless(out, atol=1e-5), f"Failed traceless for {i}"


def test_tp_even_normalization():
    a = torch.tensor([1.0, 2.0, 3.0])
    b = torch.tensor([3.2, 2.5, 1.5])
    a = a / a.norm()
    b = b / b.norm()
    a_dot_b = torch.dot(a, b)

    X = get_nt_from_vector(a, 3, normalize="unity")
    Y = get_nt_from_vector(a, 4, normalize="unity")

    for n in [1, 3, 5, 7]:
        Z = tp_even(X, Y, out_rank=n, normalize="unity")

        # check n-contraction between Z and b is equal to Legendre(a\dot b)
        indices = letter_index(n)
        rule = "".join(indices) + "," + ",".join(indices)
        tp = torch.einsum(rule, Z, *([b] * n))
        leg = legendre(a_dot_b, n)
        assert torch.allclose(tp, leg), f"Failing for n = {n}"

        # check n-contraction between Z and a is equal to 1
        tp = torch.einsum(rule, Z, *([a] * n))
        assert torch.allclose(tp, torch.tensor(1.0)), f"Failing for n = {n}"


def test_tp_odd_normalization():
    a = torch.tensor([1.0, 2.0, 3.0])
    b = torch.tensor([1.1, 2.0, 3.0])
    a = a / a.norm()
    b = b / b.norm()
    a_dot_b = torch.dot(a, b)
    a_cross_b_norm = torch.cross(a, b).norm()

    X = get_nt_from_vector(a, 3, normalize="unity")
    Y = get_nt_from_vector(b, 4, normalize="unity")

    for n in [2, 4, 6]:
        Z = tp_odd(X, Y, out_rank=n, normalize="unity")

        # Check (n-1)-contraction between Z and a, dividing a cross b is equal to
        # to Legendre(a\dot b)
        indices = letter_index(n)
        rule = "".join(indices) + "," + ",".join(indices[:-1]) + "->" + indices[-1]
        tp = torch.einsum(rule, Z, *([b] * (n - 1)))

        tp_norm = tp.norm()
        leg = legendre(a_dot_b, n)
        mul = leg * a_cross_b_norm

        assert torch.allclose(tp_norm, mul, atol=1e-3), f"Failing for n = {n}"
