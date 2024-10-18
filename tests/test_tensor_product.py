from math import factorial as factorial_math

from carten.tensor_product import (
    double_factorial,
    factorial,
    get_permutations,
    tp_even,
    tp_odd,
    tp_rule_even,
    tp_rule_odd,
)
from carten.utils import check_symmetric_traceless


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


def test_get_permutations():
    perms = get_permutations("aaabb")

    ref = [
        (0, 1, 2, 3, 4),
        (0, 1, 3, 2, 4),
        (0, 1, 3, 4, 2),
        (0, 3, 1, 2, 4),
        (0, 3, 1, 4, 2),
        (0, 3, 4, 1, 2),
        (3, 0, 1, 2, 4),
        (3, 0, 1, 4, 2),
        (3, 0, 4, 1, 2),
        (3, 4, 0, 1, 2),
    ]

    assert perms == ref


def test_tp_rule_even():
    rule, symmetry = tp_rule_even(4, 4, 4, 0)
    assert rule == "abcd,abcd->"
    assert symmetry == ""

    rule, symmetry = tp_rule_even(4, 4, 3, 0)
    assert rule == "abcd,abce->de"
    assert symmetry == "xy"

    rule, symmetry = tp_rule_even(4, 4, 3, 1)
    assert rule == "abcd,abcd,ef->ef"
    assert symmetry == "aa"

    rule, symmetry = tp_rule_even(4, 4, 2, 0)
    assert rule == "abcd,abef->cdef"
    assert symmetry == "xxyy"

    rule, symmetry = tp_rule_even(4, 4, 2, 1)
    assert rule == "abcd,abce,fg->defg"
    assert symmetry == "xyaa"

    rule, symmetry = tp_rule_even(4, 4, 2, 2)
    assert rule == "abcd,abcd,ef,gh->efgh"
    assert symmetry == "aabb"


def test_tp_rule_odd():
    rule, symmetry = tp_rule_odd(4, 4, 3, 0)
    assert rule == "abc,bdef,cdef->a"
    assert symmetry == "x"

    rule, symmetry = tp_rule_odd(4, 4, 2, 0)
    assert rule == "abc,bdef,cdeg->afg"
    assert symmetry == "xyz"

    rule, symmetry = tp_rule_odd(4, 4, 2, 1)
    assert rule == "abc,bdef,cdef,gh->agh"
    assert symmetry == "xaa"


def test_tp_even(NT3, NT4):
    for i in [0, 2, 4, 6]:
        out = tp_even(NT3, NT3, out_rank=i)
        assert out.dim() == i
        check_symmetric_traceless(out)

    for i in [0, 2, 4, 6, 8]:
        out = tp_even(NT4, NT4, out_rank=i)
        assert out.dim() == i
        check_symmetric_traceless(out)

    for i in [1, 3, 5, 7]:
        out = tp_even(NT3, NT4, out_rank=i)
        assert out.dim() == i
        check_symmetric_traceless(out)


def test_tp_odd(NT3, NT4):
    for i in [2, 4, 6]:
        out = tp_odd(NT3, NT4, out_rank=i)
        assert out.dim() == i
        check_symmetric_traceless(out)
