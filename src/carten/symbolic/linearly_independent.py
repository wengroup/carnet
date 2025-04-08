"""
Find linearly independent natural tensors from general tensors.

Reference:

References:
1. [CS70] Irreducible Cartesian Tensors. II. General Formulation, http://dx.doi.org/10.1063/1.1665190
2. [AG82] Irreducible fourth-rank Cartesian tensors, https://doi.org/10.1103/PhysRevA.25.2647
"""

import itertools
from collections import Counter, defaultdict
from fractions import Fraction
from functools import reduce
from math import gcd
from pprint import pprint

import torch
from torch import Tensor

from carten.core.reduce import symmetrize_and_remove_trace
from carten.core.symmetrize import get_permutations_2
from carten.core.utils import (dij, eijk, is_symmetric, is_traceless,
                               letter_index)
from carten.symbolic.symbolic_tensor import (CartesianTensor, Delta, Epsilon,
                                             LinearCombination, Scalar,
                                             TensorProduct, multiply_2,
                                             simplify_2)
from carten.symbolic.utils import find_independent_tensors, matrix_inverse


def get_E(j: int, s_letters: str = None) -> LinearCombination:
    """
    Invariant tensors of rank j: E(j | j).

    References:
        Eq 19 and Eq 21 of [CS70].

    Args:
        j: rank of the projection operator
        s_letters: letters for the upper case indices, if None, use the default:
            A, B, C, etc.

    Returns:
        Tensor operations with delta tensors. We use lower case letters `a`, `b`, `c`,
        etc. for indices `r`, and upper case letters `A`, `B`, `C`, etc. for indices `s`.
    """
    k = j // 2

    out = []
    c = 1  # c for t = 0
    for t in range(k + 1):
        if t > 0:
            c = -c * Fraction(
                (j - 2 * t + 2) * (j - 2 * t + 1), 2 * t * (2 * j - 2 * t + 1)
            )

        # get all rules
        all_rules = get_E_rules(j, t, s_letters)

        # Total factor: c * 1/len(all_rules), where 1/len(all_rules) averages over all
        # the rules.
        factor = Fraction(1, len(all_rules)) * c

        # create tensor products of deltas for each rule
        delta_tensors = [
            create_delta_tensors(rule["d_rs"] + rule["d_rr"] + rule["d_ss"], factor)
            for rule in all_rules
        ]

        out.extend(delta_tensors)

        # print(f"@@@ debug E_j: j={j}, t={t}, c={c}")

    return LinearCombination(*out)


def get_G_even(j: int, n: int) -> list[LinearCombination]:
    r"""
    Mapping operator G to map minimal rank tensor subspaces j onto the space n.

    G(n|j)^q = E_j \otimes^{n-j} f_{n-j}^q.

    This is for even n-j.

    Reference: Eq. 2.4 of [AG82].

    Args:
        j: the minimal tensor subspace
        n: the space to map to

    Returns:
        A list of Tensors objects, each corresponding to a q in f_{n-j}^q.
    """

    assert (n - j) % 2 == 0, f"n-j must be even, got n={n}, j={j}"

    E_s_letters, delta_rules = get_G_rules_even(j, n)

    all_G = []
    for si, rule in zip(E_s_letters, delta_rules):
        E_j = get_E(j, s_letters=si)
        f_q = create_delta_tensors(rule)
        G = multiply_2(E_j, f_q)
        all_G.append(G)

    return all_G


def get_G_odd(j: int, n: int) -> list[LinearCombination]:
    r"""
    Mapping operator G to map minimal rank tensor subspaces j onto the space n.


    G(n|j)^q = E_j \otimes^{n-j} f_{n-j}^q.

    This is for odd n-j.

    Reference: Eq. 2.5 of [AG82].

    Args:
        j: the minimal tensor subspace
        n: the space to map to

    Returns:
        A list of Tensors objects, each corresponding to a q in f_{n-j}^q.
    """
    assert (n - j) % 2 == 1, f"n-j must be odd, got n={n}, j={j}"

    E_s_letters, f_epsilon_rules, f_delta_rules = get_G_rules_odd(j, n)

    all_G = []
    for si, e_rule, d_rule in zip(E_s_letters, f_epsilon_rules, f_delta_rules):
        E_j = get_E(j, s_letters=si)
        f_q_epsilon = Epsilon(e_rule)
        f_q_delta = create_delta_tensors(d_rule)
        G = multiply_2(E_j, f_q_epsilon, f_q_delta)
        all_G.append(G)

    return all_G


def get_h_pq(g_pq: list[list[Fraction]]) -> list[list[Fraction]]:
    """
    Compute h_pq matrix, which is the inverse of g_pq.
    """
    h_pq = matrix_inverse(g_pq)

    return h_pq


def get_H(
    h_pq: list[list[Fraction]], G: list[LinearCombination]
) -> list[LinearCombination]:
    r"""
    Get the H mapping: H^p = \sum_q h_pq G^q.
    """
    H = []
    for row in h_pq:
        tensors = []
        for h_q, G_q in zip(row, G):
            if h_q == 0:
                continue
            t = multiply_2(Scalar(h_q), G_q)
            tensors.extend(t)
        H.append(LinearCombination(*tensors))

    return H


def create_delta_tensors(rule: list[str], factor: int | Fraction = 1) -> TensorProduct:
    """Create a Tensors object for deltas given the rules.

    Args:
        rule: Each string contains a pair of indices for a delta tensor.
        factor: additional factor to multiply with the tensor product

    Returns:
        List of TensorProduct objects.
    """

    tensors = [Delta(pair) for pair in rule]
    tp = TensorProduct(*tensors, factor=factor)

    return tp


def get_E_rules(j: int, t: int, s_letters: str = None) -> list[dict[str : list[str]]]:
    """
    Rules for E(j|j): d_{rs}^{j-2t} d_{rr}^t d_{ss}^t.

    This is in Eq. 19 of the paper.

    Args:
        j: rank of the projection operator
        t: number of d_rr and d_ss
        s_letters: letters for the upper case indices, if None, use the default:
            A, B, C, etc.

    Examples:
        get_E_rules(3, 1)
        [{'d_rs': ['cC'], 'd_rr': ['ab'], 'd_ss': ['AB']},
         {'d_rs': ['cB'], 'd_rr': ['ab'], 'd_ss': ['AC']},
         {'d_rs': ['cA'], 'd_rr': ['ab'], 'd_ss': ['BC']},
         {'d_rs': ['bC'], 'd_rr': ['ac'], 'd_ss': ['AB']},
         {'d_rs': ['bB'], 'd_rr': ['ac'], 'd_ss': ['AC']},
         {'d_rs': ['bA'], 'd_rr': ['ac'], 'd_ss': ['BC']},
         {'d_rs': ['aC'], 'd_rr': ['bc'], 'd_ss': ['AB']},
         {'d_rs': ['aB'], 'd_rr': ['bc'], 'd_ss': ['AC']},
         {'d_rs': ['aA'], 'd_rr': ['bc'], 'd_ss': ['BC']},
        ]

    Returns:
        Each dict {'d_rs': list_rs, 'd_rr': list_rr, 'd_ss': list_ss} contains the
        indices for constructing the deltas.

    """
    assert j >= 2 * t, f"j must be greater than or equal to 2*t, got j={j}, t={t}"

    r_letters = letter_index(j, upper_case=False)

    if s_letters is None:
        s_letters = letter_index(j, upper_case=True)

    perms = get_permutations_2(j, num_delta=t)

    # TODO, this depends on the order of the indices get_permutations_2 returns, where
    #   we put the remaining indices of t at the front, and the contracted indices at
    #   the end.
    start = j - 2 * t

    all_indices = []
    for p_r in perms:
        r_indices = [r_letters[p_r.index(i)] for i in range(j)]

        # indices for d_{rr}^t
        rr_pairs = [r_indices[i] + r_indices[i + 1] for i in range(start, j, 2)]

        # permute the remaining indices that will be used for d_{rs}^{j-2t}
        r_remaining = r_indices[:start]
        r_remaining_perms = list(itertools.permutations(r_remaining))

        for p_s in perms:
            s_indices = [s_letters[p_s.index(i)] for i in range(j)]

            # indices for d_{ss}^t
            ss_pairs = [s_indices[i] + s_indices[i + 1] for i in range(start, j, 2)]

            # permute the remaining indices that will be used for d_{rs}^{j-2t}
            s_remaining = s_indices[:start]

            # Create indices permutations for d_{rs}^{j-2t}
            for r_remaining_p in r_remaining_perms:
                rs_pairs = [f"{r}{s}" for r, s in zip(r_remaining_p, s_remaining)]

                # Sort it here to make it ordered according to the indices of r.
                # For example, ['bA', 'aB'] -> ['aB', 'bA']
                # But sort it or not does not matter for the creation of the delta
                # tensors.
                rs_pairs = sorted(rs_pairs)

                all_indices.append(
                    {"d_rs": rs_pairs, "d_rr": rr_pairs, "d_ss": ss_pairs}
                )

    return all_indices


def get_G_rules_even(j: int, n: int) -> tuple[list[str], list[list[str]]]:
    """
    Rules for G(n|j) for even n-j.

    Args:
        j:
        n:

    Returns:
        E_s_indices: s letters to use for E_j
        f_rules: rules to create deltas for for f_{n-j}^q
    """
    assert (n - j) % 2 == 0, f"n-j must be even, got n={n}, j={j}"

    letters = letter_index(n, upper_case=True)

    all_perms = get_permutations_2(n, num_delta=(n - j) // 2)

    # TODO, this depends on the order of the indices get_permutations_2 returns, where
    #   we put the remaining indices of t at the front, and the contracted indices at
    #   the end.
    start = j

    f_rules = []
    E_s_letters = []
    for perm in all_perms:  # each perm for a q in f_q
        indices = [letters[perm.index(i)] for i in range(n)]

        # indices for f_{n-j}^q
        delta_pairs = [indices[i] + indices[i + 1] for i in range(start, n, 2)]
        f_rules.append(delta_pairs)

        # s indices for E_j
        s_remaining = "".join(indices[:start])
        E_s_letters.append(s_remaining)

    return E_s_letters, f_rules


def get_G_rules_odd(j: int, n: int) -> tuple[list[str], list[str], list[list[str]]]:
    """
    Rules for G(n|j) for odd n-j.


    # NOTE,
    Upper case letter n+1 will be used as the index in epsilon to contract with E(j|j).
    In other words, it is the tau index.
    For example, if n = 3, then the letter D will be used as the tau index.

    Args:
        j:
        n:

    Returns:
        E_s_indices: s letters to use for E_j
        f_epsilon_rules: rules to create epsilons for f_{n-j}^q
        f_delta_rules: rules to create deltas for f_{n-j}^q
    """
    assert (n - j) % 2 == 1, f"n-j must be odd, got n={n}, j={j}"

    if j == 0:
        return get_G_rules_odd_j0(j, n)

    # All s letters
    letters = letter_index(n, upper_case=True)

    # Extra letter used in epsilon. See Table I of [AG82]
    tau_letter = letter_index(1, start=n, upper_case=True)

    all_perms = get_permutations_2(n, num_delta=(n - j - 1) // 2)

    # TODO, this depends on the order of the indices get_permutations_2 returns, where
    #   we put the remaining indices of t at the front, and the contracted indices at
    #   the end.
    start = j + 1

    f_delta_rules = []
    f_epsilon_rules = []
    E_s_letters = []
    for perm in all_perms:  # each perm for a q in f_q
        indices = [letters[perm.index(i)] for i in range(n)]

        # delta indices for f_{n-j}^q
        delta_pairs = [indices[i] + indices[i + 1] for i in range(start, n, 2)]

        # remaining indices for epsilon and E_j
        s_remaining = indices[:start]
        s_remaining_set = set(s_remaining)

        for comb in itertools.combinations(s_remaining, 2):
            f_delta_rules.append(delta_pairs)

            # choose two indices for epsilon
            f_epsilon_rules.append(tau_letter + "".join(sorted(comb)))

            # the remaining indices and also tau for E_j
            E_s_letters.append(
                "".join(sorted(s_remaining_set - set(comb))) + tau_letter
            )

    return E_s_letters, f_epsilon_rules, f_delta_rules


def get_G_rules_odd_j0(j, n):
    """
    For j = 0, and odd n, the rules for G(n|0) are different from the general case.

    Here we do a trivial contraction with epsilon tensor, instead of a double contraction in the general case.
    """
    assert j == 0, f"j must be 0, got {j}"
    assert n % 2 == 1, f"n must be odd, got {n}"

    # All s letters
    letters = letter_index(n, upper_case=True)

    all_perms = get_permutations_2(n, num_delta=(n - 3) // 2)

    # TODO, this depends on the order of the indices get_permutations_2 returns, where
    #   we put the remaining indices of t at the front, and the contracted indices at
    #   the end.
    start = 3

    f_delta_rules = []
    f_epsilon_rules = []
    E_s_letters = []

    for perm in all_perms:  # each perm for a q in f_q
        indices = [letters[perm.index(i)] for i in range(n)]

        # delta indices for f_{n-j}^q
        delta_pairs = [indices[i] + indices[i + 1] for i in range(start, n, 2)]
        f_delta_rules.append(delta_pairs)

        # remaining indices for epsilon (indices for E_j is empty)
        s_remaining = indices[:start]
        f_epsilon_rules.append("".join(sorted(s_remaining)))
        E_s_letters.append("")

    return E_s_letters, f_epsilon_rules, f_delta_rules


def shift_index(
    tensor: CartesianTensor | TensorProduct, shift: int, letters: str = None
) -> CartesianTensor | TensorProduct:
    """
    Shift the index of a tensor by a certain amount.

    For example, for T_ijk, and shift=1, the new tensor is T_jkl.

    Args:
        tensor: The tensor to shift the index.
        shift: The amount to shift the index.
        letters: The letters signifying the indices to shift. If None, shift all the
            indices. For example, if T_ijAB, and letters = 'AB', then the new tensor
            would be T_ijCD, where C and D are the new letters.

    Returns:
        The new tensor with the shifted index.
    """

    def _shift(t: CartesianTensor):
        if letters is None:
            indices = "".join([chr(ord(i) + shift) for i in t.indices])
        else:
            indices = "".join(
                [chr(ord(i) + shift) if i in letters else i for i in t.indices]
            )
        return t.__class__(indices, factor=t.factor, symbol=t.symbol)

    if isinstance(tensor, CartesianTensor):
        return _shift(tensor)

    elif isinstance(tensor, TensorProduct):
        components = [_shift(t) for t in tensor]
        return tensor.__class__(*components, factor=tensor.factor)

    else:
        raise ValueError(f"Unknown tensor type: {type(tensor)}")


def shift_index_2(
    tensor: LinearCombination, shift: int, letters: str = None
) -> LinearCombination:
    """
    Shift all the index of a Tensors object by a certain amount.

    Returns:
        The new tensor with the shifted index.
    """
    components = [shift_index(t, shift, letters) for t in tensor]
    return LinearCombination(*components)


def evaluate_delta(
    tensor: CartesianTensor | TensorProduct,
) -> CartesianTensor | TensorProduct:
    """Evaluate delta_ii to 3."""

    def _evaluate(t: CartesianTensor):
        if isinstance(t, Delta) and len(set(t.indices)) == 1:
            return Scalar(t.factor * 3)
        else:
            return t

    if isinstance(tensor, CartesianTensor):
        return _evaluate(tensor)

    elif isinstance(tensor, TensorProduct):
        components = [_evaluate(t) for t in tensor]
        return TensorProduct(*components, factor=tensor.factor)
    else:
        raise ValueError("Unexpected type")


def evaluate_delta_2(tensor: LinearCombination) -> LinearCombination:
    """Evaluate a linear combination of tensors."""
    components = [evaluate_delta(t) for t in tensor]
    return LinearCombination(*components)


def contract_G(
    G1: LinearCombination, G2: LinearCombination, G1_indices: str, G2_indices: str
) -> LinearCombination:
    """
    Contract two G tensors.

    Args:
        G1: The first G tensor
        G2: The second G tensor

    Returns:
        The contracted tensor.
    """
    contraction_delta = [Delta(i + j) for i, j in zip(G1_indices, G2_indices)]
    contraction_delta = TensorProduct(*contraction_delta)
    prod = multiply_2(G1, G2, contraction_delta)
    simplified = simplify_2(prod)

    return simplified


# TODO, this, and the below few functions, has been reimplemented in
#  TensorProduct.canonize()
def canonize_delta_indices(
    tensor: CartesianTensor | TensorProduct,
) -> CartesianTensor | TensorProduct:
    """
    Let the indices of delta tensors be sorted.

    For example, delta_ab -> delta_ab, delta_ba -> delta_ab.
    """

    def _canonize(t: CartesianTensor):
        if isinstance(t, Delta):
            indices = "".join(sorted(t.indices))
            return Delta(indices, factor=t.factor)
        else:
            return t

    if isinstance(tensor, CartesianTensor):
        return _canonize(tensor)
    elif isinstance(tensor, TensorProduct):
        components = [_canonize(t) for t in tensor]
        return TensorProduct(*components, factor=tensor.factor)
    else:
        raise ValueError("Unexpected type")


def canonize_delta_indices_2(tensor: LinearCombination) -> LinearCombination:
    components = [canonize_delta_indices(t) for t in tensor]
    return LinearCombination(*components)


def order_tp_components(tp: TensorProduct) -> TensorProduct:
    """Order the components of tensor product according to string representation.

    For example,
    delta_ab delta_cd -> delta_ab delta_cd
    delta_cd delta_ab -> delta_ab delta_cd
    """
    # the tensor product is just a scalar factor
    if len(tp) == 0:
        return tp

    symbols = [t.symbol for t in tp]
    indices = [t.indices for t in tp]

    str_rep = [f"{s}_{i}" for s, i in zip(symbols, indices)]

    # sort the components according to the sorted string representation
    sorted_comp, _ = zip(*sorted(zip(tp.components, str_rep), key=lambda x: x[1]))

    # create new to using the sorted symbols and indices
    new_tp = TensorProduct(*sorted_comp, factor=tp.factor)

    return new_tp


def order_tp_components_2(tensor: LinearCombination) -> LinearCombination:
    """Order the components of tensor product according to string representation."""
    out = []
    for t in tensor:
        if isinstance(t, TensorProduct):
            out.append(order_tp_components(t))
        else:
            out.append(t)
    return LinearCombination(*out)


def combine_terms(tensor: LinearCombination) -> LinearCombination:
    """
    Combine terms with the same indices.

    TODO, This currently only works for delta tensors.
    """

    # def is_delta_tp_equal(tp1: TensorProduct, tp2: TensorProduct) -> bool:
    #     """
    #     Check if tow tensor products made only of delta tensors are equal.
    #
    #     For example, delta_ab delta_cd == delta_cd delta_ab.
    #     """
    #     if len(tp1) != len(tp2):
    #         return False
    #
    #     # check the set of indices are the same
    #     return {t.indices for t in tp1} == {t.indices for t in tp2}

    def get_str_rep(tp: TensorProduct) -> str:
        """
        Get a string representation of a tensor product.

        Does not consider factor.

        For example, delta_ab delta_cd -> "ab-cd".
        """
        return "-".join(sorted(t.indices for t in tp))

    def combine(*tps: TensorProduct) -> TensorProduct:
        """Combine multiple equal tensor products."""
        factor = sum(t.factor for t in tps)
        return TensorProduct(*(tps[0]), factor=factor)

    # group the tensors with the same indices
    grouped = defaultdict(list)
    for t in tensor:
        grouped[get_str_rep(t)].append(t)

    # we loop over sorted keys to make the order deterministic
    all_combined = []
    for rep in sorted(grouped.keys()):
        tps = grouped[rep]
        if len(tps) > 1:
            all_combined.append(combine(*tps))
        else:
            all_combined.extend(tps)

    return LinearCombination(*all_combined)


def scalar_factor(t1: LinearCombination, t2: LinearCombination) -> Fraction | None:
    """
    Check if two tensors are scalar multiples of each other.

    Namely, t1 = c * t2, where c is a scalar.

    Args:
        t1:
        t2:

    Returns:
        The scalar factor to multiple with t2 to get t1. If None, t1 is not a scalar
        multiple of t2.
    """

    def to_dict(tensor: LinearCombination):
        d = {}
        for tp in tensor:
            k = " ".join([f"{t.symbol}_{t.indices}" for t in tp])
            d[k] = tp.factor
        return d

    t1_d = to_dict(t1)
    t2_d = to_dict(t2)

    if t1_d.keys() == t2_d.keys():
        factors = [t1_d[k] / t2_d[k] for k in t1_d.keys()]
        # Check all the factors are the same
        if len(set(factors)) == 1:
            return factors[0]
        else:
            raise ValueError(
                "Tensor 1 is not a scale multiple of tensor 2, although they have "
                "the same components Symbols."
            )

    else:
        return None


def get_g_pq(
    j: int, n: int, G_p: LinearCombination, G_q: LinearCombination
) -> Fraction | None:
    r"""
    Compute a single g_pq value, which is defined as
     G^p(n|j} \odot^n G^q(n|j) = g_pq E(j|j).

    Args:
        j:

    Returns:
        The scalar factor g_pq. If None, G_p and G_q are not scalar multiples of each
        other.
    """
    even = (n - j) % 2 == 0

    # G_p and G_q are using the same set of indices. Should shift one of them to carry
    # out the contraction.
    if even:
        shift = n
    else:
        if j == 0:
            shift = n  # there is no tau
        else:
            shift = n + 1  # the additional one for tau
    G_q = shift_index_2(G_q, shift)

    def get_upper_indices(G: LinearCombination) -> str:
        letters = set()
        for t in G:
            letters.update([i for i in t.indices if i.isupper()])
        return "".join(sorted(letters))

    # Contracted indices s1, s2, ..., sn of G_p and G_q (upper case letters)
    p_idx = get_upper_indices(G_p)
    q_idx = get_upper_indices(G_q)

    # For odd n-j and j is not 0, the index tau should not be contracted.
    # It is the latest upper case letter, see get_G_rules_odd().
    # For odd n-j, and j is 0, there is no tau index.
    if not even and j != 0:
        p_idx = p_idx[:-1]
        q_idx = q_idx[:-1]

    contracted = contract_G(G_p, G_q, p_idx, q_idx)
    contracted = combine_terms(
        order_tp_components_2(canonize_delta_indices_2(evaluate_delta_2(contracted)))
    )

    # in the contracted tensor, all upper case indices s1, ... sn are contracted.
    # To ensure contracted and E_jj using the same set of indices, we provide the
    # remaining indices in G_q to E_jj.
    remaining_indices = set()
    for t in G_q:
        remaining_indices.update([i for i in t.indices if i.islower()])
    remaining_indices = "".join(sorted(remaining_indices))

    E_jj = get_E(j, remaining_indices)
    E_jj = order_tp_components_2(E_jj)

    # Compare contracted and E_jj to get the factor  g_pq

    factor = scalar_factor(contracted, E_jj)

    return factor


def get_g_pq_matrix(
    j: int, n: int, all_G: list[LinearCombination]
) -> list[list[Fraction]]:
    """
    Compute a matrix of g_pq values.
    Args:
        j:
        n:
        all_G:

    Returns:
    """
    num = len(all_G)

    matrix = [[None] * num for _ in range(num)]
    for p in range(num):
        for q in range(num):
            v = get_g_pq(j, n, all_G[p], all_G[q])
            # If there is no scalar factor, setting it to 0.
            if v is None:
                v = Fraction(0)
            matrix[p][q] = v

    return matrix


def embed(j: int, G: LinearCombination, X: Tensor = None, seed: int = 35) -> Tensor:
    r"""
    Evaluate S(n) = G(n|j) \odot^n X(j).

    Recall, in G, lower case indices are for r1, r2, ..., rj, and upper case indices
    are for s1, s2, ..., sn.

    Args:
        G: the contraction rule.
        X: the natural tensor X(j) to contract with G. If None, a random one is created.

    Return:
        S(n) in the space n.
    """

    if X is None:
        torch.manual_seed(seed)
        X = torch.randn(3**j).reshape([3] * j)
        X = symmetrize_and_remove_trace(X)

    d = dij()
    e = eijk()

    output = []
    for tp in G:
        # create contraction rule
        indices = [t.indices for t in tp]
        delta_epsilon_rule = ",".join(indices)
        X_rule = "".join(sorted([i for i in "".join(indices) if i.islower()]))

        # For odd n-j and j != 0 the index for tau will appear twice, they should be
        # removed for the S rule.
        upper = "".join([i for i in "".join(indices) if i.isupper()])
        S_rule = "".join(sorted([s for s, n in Counter(upper).items() if n == 1]))

        rule = f"{delta_epsilon_rule},{X_rule}->{S_rule}"

        # get delta and epsilon tensors for contraction
        delta_epsilon = []
        seen_epsilon = False
        for comp in tp:
            if isinstance(comp, Delta):
                delta_epsilon.append(d)
            elif isinstance(comp, Epsilon):
                if seen_epsilon:
                    raise ValueError("Only one epsilon tensor is allowed.")
                else:
                    seen_epsilon = True
                delta_epsilon.append(e)
            else:
                raise ValueError("Unexpected type.")

        # TODO, the rules for tensor product epsilons and deltas can be precomputed and
        #  summed up. Then, we only need a single contraction.
        # perform the contraction
        S = float(tp.factor) * torch.einsum(rule, *delta_epsilon, X)
        output.append(S)

    return torch.stack(output).sum(dim=0)


def extract(H: LinearCombination, T: Tensor = None) -> Tensor:
    r"""
    Evaluate X^p,j = H^p(j|n) \odot^n T(n).

    Args:
        H:
        T:

    Returns:
    """

    d = dij()
    e = eijk()

    output = []
    for tp in H:
        # create contraction rule
        indices = [t.indices for t in tp]
        delta_epsilon_rule = ",".join(indices)
        X_rule = "".join(sorted([i for i in "".join(indices) if i.islower()]))

        # For odd n-j and j!=0, the index for tau will appear twice, they should be
        # removed for the T rule.
        upper = "".join([i for i in "".join(indices) if i.isupper()])
        T_rule = "".join(sorted([s for s, n in Counter(upper).items() if n == 1]))

        rule = f"{delta_epsilon_rule},{T_rule}->{X_rule}"

        # get delta and epsilon tensors for contraction
        delta_epsilon = []
        seen_epsilon = False
        for comp in tp:
            if isinstance(comp, Delta):
                delta_epsilon.append(d)
            elif isinstance(comp, Epsilon):
                if seen_epsilon:
                    raise ValueError("Only one epsilon tensor is allowed.")
                else:
                    seen_epsilon = True
                delta_epsilon.append(e)
            else:
                # tp only consists of delta and epsilon tensors
                raise ValueError(f"Unexpected type. {type(comp)}")

        # TODO, the rules tensor product epsilons and deltas can be precomputed and
        #  summed up. Then, we only need a single contraction.
        #
        # perform the contraction
        X = float(tp.factor) * torch.einsum(rule, *delta_epsilon, T)
        output.append(X)

    return torch.stack(output).sum(dim=0)


def check_one(prod):
    evaluated = evaluate_delta_2(prod)
    print("@@@ evaluated:", evaluated)

    canolized = canonize_delta_indices_2(evaluated)
    print("@@@ canolized:", canolized)

    ordered = order_tp_components_2(canolized)
    print("@@@ ordered:", ordered)

    combined = combine_terms(ordered)
    print("@@@ combined:", combined)


def find_matrix_factorization(
    A: list[list[Fraction]],
) -> tuple[Fraction, list[list[int]]]:
    """
    For a matrix A consisting of Fraction, find the scalar factor c and integer
    matrix B such that A = c*B.
    """
    # Flatten and get nums/denoms
    nums = [f.numerator for row in A for f in row if f is not None]
    denoms = [f.denominator for row in A for f in row if f is not None]

    # Find GCD of nums and LCM of denoms
    def lcm(a, b):
        return abs(a * b) // gcd(a, b)

    num_gcd = reduce(gcd, [abs(n) for n in nums])
    denom_lcm = reduce(lcm, denoms)

    # Scalar factor
    c = Fraction(num_gcd, denom_lcm)

    # Integer matrix B = A/c
    B = [[int(f / c) if f is not None else None for f in row] for row in A]

    return c, B


if __name__ == "__main__":
    # ################################################################################
    # odd n-j
    # j = 2
    # n = 3
    # all_G = get_G_odd(j, n)

    # G1 = all_G[0]
    # G2 = all_G[1]
    # G3 = all_G[2]
    # print("G_1:", G1)
    # print("G_2:", G2)
    # print("G_3:", G3)

    # G1_shifted = shift_index_2(G1, shift=12)
    # G2_shifted = shift_index_2(G2, shift=4)
    # G3_shifted = shift_index_2(G3, shift=8)
    # print("G_1, after shift:", G1_shifted)
    # print("G_2, after shift:", G2_shifted)
    # print("G_3, after shift:", G3_shifted)
    #
    # # check they are linearly independent
    # prod = contract_G(G1, G2_shifted, "ABC", "EFG")
    # print("=" * 40)
    # check_one(prod)
    #
    # prod = contract_G(G1, G3_shifted, "ABC", "IJK")
    # print("=" * 40)
    # check_one(prod)
    #
    # prod = contract_G(G2_shifted, G3_shifted, "EFG", "IJK")
    # print("=" * 40)
    # check_one(prod)
    #
    # prod = contract_G(G1, G1_shifted, "ABC", "MNO")
    # print("=" * 40)
    # check_one(prod)

    ################################################################################
    # # even n-j
    # j = 4
    # n = 4
    # all_G = get_G_even(j, n)
    #
    # print("number of G", len(all_G))

    # G1 = all_G[0]
    # G2 = all_G[1]
    # G3 = all_G[2]
    # print("G_1:", G1)
    # print("G_2:", G2)
    # print("G_3:", G3)
    #
    # G1_shifted = shift_index_2(G1, shift=12)
    # G2_shifted = shift_index_2(G2, shift=4)
    # G3_shifted = shift_index_2(G3, shift=8)
    # print("G_1, after shift:", G1_shifted)
    # print("G_2, after shift:", G2_shifted)
    # print("G_3, after shift:", G3_shifted)
    #
    # # check they are linearly independent
    # prod = contract_G(G1, G2_shifted, "ABCD", "EFGH")
    # print("=" * 40)
    # check_one(prod)
    #
    # prod = contract_G(G1, G3_shifted, "ABCD", "IJKL")
    # print("=" * 40)
    # check_one(prod)
    #
    # prod = contract_G(G2_shifted, G3_shifted, "EFGH", "IJKL")
    # print("=" * 40)
    # check_one(prod)
    #
    # prod = contract_G(G1, G1_shifted, "ABCD", "MNOP")
    # print("=" * 40)
    # check_one(prod)

    ################################################################################

    def extract_and_embed(j: int, n: int, T: Tensor):
        """
        Extract the natural tensor X(j) from T(n) and embed it back to get S(n).

        Args:
            j: weight
            n: dim of the space T is in
            T: A general tensor in space n.

        Returns:
            All X^p,j tensors, which are the natural tensors in space j.
            All embedding tensors S^p(n), whose corresponding G^p(n|j) are linearly
            independent.
        """
        print("=" * 80)
        print(f"j={j}, n={n}")

        # create G mapping operator
        if (n - j) % 2 == 0:
            all_G = get_G_even(j, n)
        else:
            all_G = get_G_odd(j, n)
        print("Number of candidate G:", len(all_G))

        # Determine g_pq for all G
        # TODO, this block can be removed (not needed for selecting the independent ones)
        g_pq = get_g_pq_matrix(j, n, all_G)
        if len(all_G) > 1:
            c, g_pq_int = find_matrix_factorization(g_pq)
        else:
            c = 1
            g_pq_int = g_pq
        print("g_pq matrix for all G:")
        print("c:", c)
        print("matrix:")
        pprint(g_pq_int)

        # Get S tensors, embedding space j to space n
        all_S = [embed(j, G) for G in all_G]

        # Get linearly independent G tensors
        _, independent_indices = find_independent_tensors(all_S)

        # TODO, for tensors will partial symmetry, we can implement additional rules
        #  to filter out some of all_G by considering the symmetry of the tensor T.

        independent_G = [all_G[i] for i in independent_indices]

        print("Number of independent G:", len(independent_G))
        print("Selected independent indices:", independent_indices)
        for p, G in enumerate(independent_G):
            print(f"p={p}, G=")
            print(G)

        # Get g_pq matrix for independent G
        g_pq = get_g_pq_matrix(j, n, independent_G)
        if len(independent_G) > 1:
            # Note, c may not be 1, but g_pq should consist of the factor c
            # Here we use g_pq_int just for nice printing and visualization.
            # And we should use g_pq for the actual computation.
            c, g_pq_int = find_matrix_factorization(g_pq)
        else:
            c = 1
            g_pq_int = g_pq
        print("g_pq matrix for independent ones:")
        print("c:", c)
        print("matrix:")
        pprint(g_pq_int)

        # Get h_pq matrix
        h_pq = get_h_pq(g_pq)
        print("h_pq:")
        pprint(h_pq)

        # Get H tensors, extracting from the space n to space j
        all_H = get_H(h_pq, independent_G)
        for p, H in enumerate(all_H):
            print(f"p={p}, H=")
            print(H)

        ########################################
        # symbolic S
        for i, (G, H) in enumerate(zip(independent_G, all_H)):
            # Shift Upper letters of H to distinguish those from G
            H = shift_index_2(H, n, letter_index(24, upper_case=True))

            G = simplify_2(G)
            H = simplify_2(H)
            S = multiply_2(G, H)
            S = simplify_2(S)

            print("=" * 10)
            print(f"G={G}")
            print(f"H={H}")
            print(f"S tensor ``S=G \odot^j H`` ({i})", S)

        ########################################
        # numerical S

        # Extracting X from T
        all_X = [extract(H, T) for H in all_H]

        for X in all_X:
            assert is_symmetric(X), f"X={X} is not symmetric"
            assert is_traceless(X), f"X={X} is not traceless"

        # Embed X back to space n
        all_S = [embed(j, G, X) for G, X in zip(independent_G, all_X)]

        return all_S

    ################################################################################
    # Extract and re-embed, and then check the sum of S (from different weight j and
    # seniority n) is equal to T.
    ################################################################################

    # create a T of rank n
    n = 3
    torch.manual_seed(35)
    T = torch.randn(3**n).reshape([3] * n)

    # check the sum of all extract and embed is equal to the original T
    all_S = []
    for j in range(n + 1):
        S_j = extract_and_embed(j, n, T)
        all_S.extend(S_j)

    sum_S = torch.stack(all_S).sum(dim=0)

    print("=" * 40)
    if torch.allclose(sum_S, T, atol=1e-5, rtol=1e-5):
        print("The sum of S is equal to T")
        print("Sum of diff:", (sum_S - T).sum())
        print("Mean absolute diff:", torch.abs((sum_S - T)).mean())
    else:
        print("The sum of S is not equal to T")
        print("Sum of S:", sum_S.sum())
        print("Sum of T:", T.sum())
        print("Sum of diff:", (sum_S - T).sum())
        print("Mean absolute diff:", torch.abs((sum_S - T)).mean())
