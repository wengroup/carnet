import torch

from carten.tensor_product import get_tp_even_rule, get_tp_odd_rule, tp_even, tp_odd
from carten.utils import is_symmetric, is_symmetric_traceless, is_traceless


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
    v = torch.tensor([1.0, 2.0, 3.0])
    unit_vector = v / v.norm()

    for i in [1, 3, 5, 7]:
        out = tp_even(NT3, NT4, out_rank=i, normalize="unity")
        assert out.ndim == i
        assert is_symmetric(out, atol=1e-6), f"Not symmetric for {i}"
        assert is_traceless(out, atol=1e-5), f"Not traceless for {i}"

        # test unity
        # contraction with a unit vector should give 1
        tp = torch.eimsum("i,i", out, unit_vector)
        assert tp == 1.0, f"Failed unity for {i}"

    for i in [0, 2, 4, 6, 8]:
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
