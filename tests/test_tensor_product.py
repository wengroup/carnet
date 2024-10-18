from carten.tensor_product import get_permutations, tp_rule_even


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
    assert symmetry == "ab"

    rule, symmetry = tp_rule_even(4, 4, 3, 1)
    assert rule == "abcd,abcd,ef->ef"
    assert symmetry == "cc"

    rule, symmetry = tp_rule_even(4, 4, 2, 0)
    assert rule == "abcd,abef->cdef"
    assert symmetry == "aabb"

    rule, symmetry = tp_rule_even(4, 4, 2, 1)
    assert rule == "abcd,abce,fg->defg"
    assert symmetry == "abcc"

    rule, symmetry = tp_rule_even(4, 4, 2, 2)
    assert rule == "abcd,abcd,ef,gh->efgh"
    assert symmetry == "ccdd"
