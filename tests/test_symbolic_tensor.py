import itertools
from fractions import Fraction

from pytest import fixture

from carten.symbolic_tensor import (
    CartesianTensor,
    Delta,
    Epsilon,
    Scalar,
    TensorProduct,
    LinearCombination,
    Zero,
    contract_epsilon_delta,
    contract_two_epsilon,
    contract_with_delta,
    simplify,
    symmetrize,
)


@fixture
def T4():
    return CartesianTensor("ijkl")


@fixture
def delta_ij():
    return Delta("ij")


@fixture
def delta_ik():
    return Delta("ik")


@fixture
def delta_ai():
    return Delta("ai")


@fixture
def epsilon_ijk():
    return Epsilon("ijk")


@fixture
def epsilon_kij():
    return Epsilon("kij")


@fixture
def epsilon_jik():
    return Epsilon("jik")


@fixture
def epsilon_ijl():
    return Epsilon("ijl")


@fixture
def epsilon_ilm():
    return Epsilon("ilm")


def test_cartesian_tensor():
    assert CartesianTensor("ijjl") == CartesianTensor("ikkl")


def test_delta():
    assert Delta("ij") == Delta("ji")
    assert Delta("ij") != Delta("ik")


def test_epsilon(epsilon_ijk, epsilon_kij, epsilon_jik):
    assert epsilon_ijk == epsilon_kij
    assert epsilon_ijk != epsilon_jik


def test_contract_with_delta(T4, delta_ij, delta_ik, delta_ai):
    assert contract_with_delta(delta_ij, T4) == CartesianTensor("jjkl")
    assert contract_with_delta(delta_ik, T4) == CartesianTensor("kjkl")
    assert contract_with_delta(delta_ai, T4) == CartesianTensor("ajkl")

    assert contract_with_delta(delta_ij, delta_ik) == Delta("jk")


def test_contract_epsilon_delta(epsilon_ijk, delta_ij, delta_ai):
    assert contract_epsilon_delta(epsilon_ijk, delta_ij) == Zero()
    assert contract_epsilon_delta(epsilon_ijk, delta_ai) == Epsilon("ajk")


def test_contract_two_epsilon(epsilon_ijk, epsilon_ijl, epsilon_ilm):
    assert contract_two_epsilon(epsilon_ijl, epsilon_ijl) == Scalar(6)
    assert contract_two_epsilon(epsilon_ijk, epsilon_ijl) == Delta("kl", factor=2)
    assert contract_two_epsilon(epsilon_ijk, epsilon_ilm) == LinearCombination(
        TensorProduct(Delta("jl"), Delta("km")),
        TensorProduct(Delta("jm"), Delta("kl"), factor=-1),
    )


def test_symmetrize():
    indices = "ijkl"  # symmetrizing indices
    tensors = [
        CartesianTensor("".join(p), factor=Fraction(1, 24))
        for p in itertools.permutations(indices)
    ]
    assert symmetrize(CartesianTensor(indices)) == LinearCombination(*tensors)

    indices = "akl"  # symmetrizing indices
    tensors = []
    for p in itertools.permutations(indices):
        t = TensorProduct(
            Epsilon(f"{p[0]}ij"),
            CartesianTensor(f"ij{p[1]}{p[2]}"),
            factor=Fraction(1, 6),
        )
        tensors.append(t)
    lin_comb = LinearCombination(*tensors)

    t = TensorProduct(Epsilon("aij"), CartesianTensor("ijkl"))
    s = symmetrize(t)
    assert s == lin_comb


def test_simplify():
    d1 = Delta("ij", factor=2)
    d2 = Delta("jk", factor=2)
    e1 = Epsilon("ijk", factor=3)
    e2 = Epsilon("ikl", factor=3)
    e3 = Epsilon("ilm", factor=3)
    T1 = CartesianTensor("ijkl", factor=4)

    tp = TensorProduct(d1, d2)
    tp_s = simplify(tp)
    assert tp_s.to_str_list() == ["(4) δ_ik"]

    tp = TensorProduct(d1, e1)
    tp_s = simplify(tp)
    assert tp_s[0] == TensorProduct(Zero())

    tp = TensorProduct(d1, e2)
    tp_s = simplify(tp)
    assert tp_s.to_str_list() == ["(6) ε_jkl"]

    tp = TensorProduct(d2, e2)
    tp_s = simplify(tp)
    assert tp_s.to_str_list() == ["(6) ε_ijl"]

    tp = TensorProduct(e1, e2)
    tp_s = simplify(tp)
    assert tp_s.to_str_list() == ["(-18) δ_jl"]

    tp = TensorProduct(e1, e3)
    tp_s = simplify(tp)
    assert tp_s.to_str_list() == ["(9) δ_jl δ_km", "(-9) δ_jm δ_kl"]

    tp = TensorProduct(d1, e1, e3)
    tp_s = simplify(tp)
    assert tp_s.to_str_list() == ["(18) δ_il δ_km", "(-18) δ_im δ_kl"]

    tp = TensorProduct(d1, T1)
    tp_s = simplify(tp)
    assert tp_s.to_str_list() == ["(8) T_jjkl"]

    tp = TensorProduct(d1, e1, e2, T1)
    tp_s = simplify(tp)
    assert tp_s.to_str_list() == ["(-144) T_ljkl"]

    tp = TensorProduct(d1, e1, e3, T1)
    tp_s = simplify(tp)
    assert tp_s.to_str_list() == ["(72) T_ljml", "(-72) T_mjll"]
