import torch

from carten.core.Legendre import legendre
from carten.core.unit_vector import get_nt_from_vector, get_nt_from_vector_rule
from carten.core.utils import is_symmetric_traceless, letter_index


def test_get_nt_from_vector_rule():
    rule, symmetry, delta_indices = get_nt_from_vector_rule(3, 0)
    assert rule == "a,b,c->abc"
    assert symmetry == "xxx"
    assert delta_indices == ""

    rule, symmetry, delta_indices = get_nt_from_vector_rule(3, 1)
    assert rule == "a,bc->abc"
    assert symmetry == "xaa"
    assert delta_indices == "a"

    rule, symmetry, delta_indices = get_nt_from_vector_rule(4, 0)
    assert rule == "a,b,c,d->abcd"
    assert symmetry == "xxxx"
    assert delta_indices == ""

    rule, symmetry, delta_indices = get_nt_from_vector_rule(4, 1)
    assert rule == "a,b,cd->abcd"
    assert symmetry == "xxaa"
    assert delta_indices == "a"

    rule, symmetry, delta_indices = get_nt_from_vector_rule(4, 2)
    assert rule == "ab,cd->abcd"
    assert symmetry == "aabb"
    assert delta_indices == "ab"


def test_get_nt_from_vector():
    a = torch.tensor([1.0, 2.0, 3.0])
    b = torch.tensor([3.2, 2.5, 1.5])
    a = a / a.norm()
    b = b / b.norm()
    a_dot_b = torch.dot(a, b)

    nt = get_nt_from_vector(a, 0)
    assert nt == torch.tensor(1.0)

    nt = get_nt_from_vector(a, 1)
    assert torch.allclose(nt, a)

    for n in range(2, 8):
        nt = get_nt_from_vector(a, n)
        assert nt.shape == tuple([3] * n)
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
