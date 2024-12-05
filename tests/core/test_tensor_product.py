import torch

from carten.core.legendre import legendre
from carten.core.tensor_product import (
    get_tp_even_rule,
    get_tp_odd_rule,
    tp_even,
    tp_even_simple,
    tp_odd,
    tp_odd_simple,
)
from carten.core.unit_vector import get_nt_from_vector, letter_index
from carten.core.utils import is_symmetric, is_traceless


def test_tp_rule_even():
    rule, symmetry, _ = get_tp_even_rule(4, 4, 4, 0)
    assert rule == "...xabcd,...yabcd->...xy"
    assert symmetry == ""

    rule, symmetry, _ = get_tp_even_rule(4, 4, 3, 0)
    assert rule == "...xabcd,...yabce->...xyde"
    assert symmetry == "ab"

    rule, symmetry, _ = get_tp_even_rule(4, 4, 3, 1)
    assert rule == "...xabcd,...yabcd,AB->...xyAB"
    assert symmetry == "AA"

    rule, symmetry, _ = get_tp_even_rule(4, 4, 2, 0)
    assert rule == "...xabcd,...yabef->...xycdef"
    assert symmetry == "aabb"

    rule, symmetry, _ = get_tp_even_rule(4, 4, 2, 1)
    assert rule == "...xabcd,...yabce,AB->...xydeAB"
    assert symmetry == "abAA"

    rule, symmetry, _ = get_tp_even_rule(4, 4, 2, 2)
    assert rule == "...xabcd,...yabcd,AB,CD->...xyABCD"
    assert symmetry == "AABB"


def test_tp_rule_odd():
    rule, symmetry, _ = get_tp_odd_rule(4, 4, 3, 0)
    assert rule == "uvw,...xvabc,...ywabc->...xyu"
    assert symmetry == "a"

    rule, symmetry, _ = get_tp_odd_rule(4, 4, 2, 0)
    assert rule == "uvw,...xvabc,...ywabd->...xyucd"
    assert symmetry == "abc"

    rule, symmetry, _ = get_tp_odd_rule(4, 4, 2, 1)
    assert rule == "uvw,...xvabc,...ywabc,AB->...xyuAB"
    assert symmetry == "aAA"


def test_tp_even(NT3, NT4):
    # add batch (1) and multiplicity (2) dimensions
    NT3 = torch.stack([NT3, NT3]).unsqueeze(0)
    NT4 = torch.stack([NT4, NT4]).unsqueeze(0)

    for i in [1, 3, 5, 7]:
        out = tp_even(NT3, NT4, l1=3, l2=4, l3=i, normalize="unity")
        assert out.shape == (1, 4) + (3,) * i

        # get one of them
        out1 = out[0, 0]
        assert is_symmetric(out1, atol=1e-6), f"Not symmetric for {i}"
        assert is_traceless(out1, atol=1e-5), f"Not traceless for {i}"

        # check the others are the same
        for i in range(1, 4):
            assert torch.allclose(out[0, i], out1), f"Not the same for {i}"

    # for i in [0, 2, 4, 6, 8]:
    for i in [0, 2, 4, 6]:
        out = tp_even(NT4, NT4, l1=4, l2=4, l3=i, normalize="unity")
        assert out.shape == (1, 4) + (3,) * i

        # get one of them
        out1 = out[0, 0]
        assert is_symmetric(out1, atol=1e-6), f"Not symmetric for {i}"
        assert is_traceless(out1, atol=1e-5), f"Not traceless for {i}"

        # check the others are the same
        for i in range(1, 4):
            assert torch.allclose(out[0, i], out1), f"Not the same for {i}"


def test_tp_odd(NT2, NT3, NT4):
    # add batch (1) and multiplicity (2) dimensions
    NT2 = torch.stack([NT2, NT2]).unsqueeze(0)
    NT3 = torch.stack([NT3, NT3]).unsqueeze(0)
    NT4 = torch.stack([NT4, NT4]).unsqueeze(0)

    for i in [3, 5]:
        out = tp_odd(NT2, NT4, l1=2, l2=4, l3=i, normalize="unity")
        assert out.shape == (1, 4) + (3,) * i

        # get one of them
        out1 = out[0, 0]
        assert is_symmetric(out1, atol=1e-6), f"Failed symmetry for {i}"
        assert is_traceless(out1, atol=1e-5), f"Failed traceless for {i}"

        # check the others are the same
        for i in range(1, 4):
            assert torch.allclose(out[0, i], out1), f"Not the same for {i}"

    for i in [2, 4, 6]:
        out = tp_odd(NT3, NT4, l1=3, l2=4, l3=i, normalize="unity")
        assert out.shape == (1, 4) + (3,) * i

        # get one of them
        out1 = out[0, 0]
        assert is_symmetric(out1, atol=1e-6), f"Failed symmetry for {i}"
        assert is_traceless(out1, atol=1e-5), f"Failed traceless for {i}"

        # check the others are the same
        for i in range(1, 4):
            assert torch.allclose(out[0, i], out1), f"Not the same for {i}"


def test_tp_even_normalization():
    a = torch.tensor([1.0, 2.0, 3.0])
    b = torch.tensor([3.2, 2.5, 1.5])
    a = a / a.norm()
    b = b / b.norm()
    a_dot_b = torch.dot(a, b)

    X = get_nt_from_vector(a, 3, normalize="unity")
    Y = get_nt_from_vector(a, 4, normalize="unity")

    for n in [1, 3, 5, 7]:
        Z = tp_even_simple(X, Y, l3=n, normalize="unity")

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
        Z = tp_odd_simple(X, Y, l3=n, normalize="unity")

        # Check (n-1)-contraction between Z and a, dividing a cross b is equal to
        # to Legendre(a\dot b)
        indices = letter_index(n)
        rule = "".join(indices) + "," + ",".join(indices[:-1]) + "->" + indices[-1]
        tp = torch.einsum(rule, Z, *([b] * (n - 1)))

        tp_norm = tp.norm()
        leg = legendre(a_dot_b, n)
        mul = leg * a_cross_b_norm

        assert torch.allclose(tp_norm, mul, atol=1e-3), f"Failing for n = {n}"
