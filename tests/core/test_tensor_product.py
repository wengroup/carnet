import torch

from carten.core.legendre import legendre
from carten.core.tensor_product import (
    get_tp_even_rule,
    get_tp_odd_rule,
    tp_even,
    tp_odd,
)
from carten.core.unit_vector import get_nt_from_vector, letter_index
from carten.core.utils import is_symmetric, is_traceless


def test_tp_rule_even():
    rule, symmetry, _ = get_tp_even_rule(4, 4, 4, 0)
    assert rule == "...abcd,...abcd->..."
    assert symmetry == ""

    rule, symmetry, _ = get_tp_even_rule(4, 4, 3, 0)
    assert rule == "...abcd,...abce->...de"
    assert symmetry == "ab"

    rule, symmetry, _ = get_tp_even_rule(4, 4, 3, 1)
    assert rule == "...abcd,...abcd,AB->...AB"
    assert symmetry == "AA"

    rule, symmetry, _ = get_tp_even_rule(4, 4, 2, 0)
    assert rule == "...abcd,...abef->...cdef"
    assert symmetry == "aabb"

    rule, symmetry, _ = get_tp_even_rule(4, 4, 2, 1)
    assert rule == "...abcd,...abce,AB->...deAB"
    assert symmetry == "abAA"

    rule, symmetry, _ = get_tp_even_rule(4, 4, 2, 2)
    assert rule == "...abcd,...abcd,AB,CD->...ABCD"
    assert symmetry == "AABB"


def test_tp_rule_odd():
    rule, symmetry, _ = get_tp_odd_rule(4, 4, 3, 0)
    assert rule == "uvw,...vabc,...wabc->...u"
    assert symmetry == "a"

    rule, symmetry, _ = get_tp_odd_rule(4, 4, 2, 0)
    assert rule == "uvw,...vabc,...wabd->...ucd"
    assert symmetry == "abc"

    rule, symmetry, _ = get_tp_odd_rule(4, 4, 2, 1)
    assert rule == "uvw,...vabc,...wabc,AB->...uAB"
    assert symmetry == "aAA"


def test_tp_even(NT3, NT4):
    # add batch dims
    NT3 = torch.stack([NT3, NT3])
    NT4 = torch.stack([NT4, NT4])

    NT3 = NT3.view(2, -1)
    NT4 = NT4.view(2, -1)

    for l3 in [1, 3, 5, 7]:
        out = tp_even(NT3, NT4, l1=3, l2=4, l3=l3, normalize="unity")
        assert out.shape == (2, 3**l3)

        # check the two batch elements are the same
        assert torch.allclose(out[0], out[1])

        assert is_symmetric(out[0], atol=1e-6), f"Not symmetric for {l3}"
        assert is_traceless(out[0], atol=1e-5), f"Not traceless for {l3}"

    # for i in [0, 2, 4, 6, 8]:
    for l3 in [0, 2, 4, 6]:
        out = tp_even(NT4, NT4, l1=4, l2=4, l3=l3, normalize="unity")
        assert out.shape == (2, 3**l3)

        # check the two batch elements are the same
        assert torch.allclose(out[0], out[1])

        assert is_symmetric(out[0], atol=1e-6), f"Not symmetric for {l3}"
        assert is_traceless(out[0], atol=1e-5), f"Not traceless for {l3}"


def test_tp_odd(NT2, NT3, NT4):
    # add batch dims
    NT2 = torch.stack([NT2, NT2])
    NT3 = torch.stack([NT3, NT3])
    NT4 = torch.stack([NT4, NT4])

    NT2 = NT2.view(2, -1)
    NT3 = NT3.view(2, -1)
    NT4 = NT4.view(2, -1)

    for l3 in [3, 5]:
        out = tp_odd(NT2, NT4, l1=2, l2=4, l3=l3, normalize="unity")
        assert out.shape == (2, 3**l3)

        # check the two batch elements are the same
        assert torch.allclose(out[0], out[1])

        assert is_symmetric(out[0], atol=1e-6), f"Failed symmetry for {l3}"
        assert is_traceless(out[1], atol=1e-5), f"Failed traceless for {l3}"

    for l3 in [2, 4, 6]:
        out = tp_odd(NT3, NT4, l1=3, l2=4, l3=l3, normalize="unity")
        assert out.shape == (2, 3**l3)

        # get one of them
        out1 = out[0, 0]
        assert is_symmetric(out1, atol=1e-6), f"Failed symmetry for {l3}"
        assert is_traceless(out1, atol=1e-5), f"Failed traceless for {l3}"


def test_tp_even_normalization():
    a = torch.tensor([1.0, 2.0, 3.0])
    b = torch.tensor([3.2, 2.5, 1.5])
    a = a / a.norm()
    b = b / b.norm()
    a_dot_b = torch.dot(a, b)

    X = get_nt_from_vector(a, 3, normalize="unity")
    Y = get_nt_from_vector(a, 4, normalize="unity")
    X = X.view(-1)
    Y = Y.view(-1)

    for n in [1, 3, 5, 7]:
        Z = tp_even(X, Y, l1=3, l2=4, l3=n, normalize="unity")
        Z = Z.view((3,) * n)

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
    X = X.view(-1)
    Y = Y.view(-1)

    for n in [2, 4, 6]:
        Z = tp_odd(X, Y, l1=3, l2=4, l3=n, normalize="unity")
        Z = Z.view((3,) * n)

        # Check (n-1)-contraction between Z and a, dividing a cross b is equal to
        # to Legendre(a\dot b)
        indices = letter_index(n)
        rule = "".join(indices) + "," + ",".join(indices[:-1]) + "->" + indices[-1]
        tp = torch.einsum(rule, Z, *([b] * (n - 1)))

        tp_norm = tp.norm()
        leg = legendre(a_dot_b, n)
        mul = leg * a_cross_b_norm

        assert torch.allclose(tp_norm, mul, atol=1e-3), f"Failing for n = {n}"
