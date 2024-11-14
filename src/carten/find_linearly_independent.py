"""
Find linearly independent natural tensors.

Reference:

References:
1. [CS70] Irreducible Cartesian Tensors. II. General Formulation, http://dx.doi.org/10.1063/1.1665190
2. [AG82] Irreducible fourth-rank Cartesian tensors, https://doi.org/10.1103/PhysRevA.25.2647

"""
import itertools
from collections import defaultdict
from distutils.command.check import check
from fractions import Fraction

from carten.reduce import get_permutations_2
from carten.symbolic_tensor import (
    CartesianTensor,
    Delta,
    Epsilon,
    Scalar,
    TensorProduct,
    Tensors,
    multiply_2,
    simplify_2,
)
from carten.utils import letter_index


def E(j: int, s_letters: str = None) -> Tensors:
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
    n = j // 2

    out = []
    c = 1  # c for t = 0
    for t in range(n + 1):
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

    return Tensors(*out)


def G_even(j: int, n: int) -> list[Tensors]:
    """
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
        E_j = E(j, s_letters=si)
        f_q = create_delta_tensors(rule)
        G = multiply_2(E_j, f_q)
        all_G.append(G)

    return all_G


def G_odd(j: int, n: int) -> list[Tensors]:
    """
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
        E_j = E(j, s_letters=si)
        f_q_epsilon = Epsilon(e_rule)
        f_q_delta = create_delta_tensors(d_rule)
        G = multiply_2(E_j, f_q_epsilon, f_q_delta)
        all_G.append(G)

    return all_G


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
        >>> get_E_rules(3, 1)
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

    Args:
        j:
        n:

    Returns:
        E_s_indices: s letters to use for E_j
        f_epsilon_rules: rules to create epsilons for f_{n-j}^q
        f_delta_rules: rules to create deltas for for f_{n-j}^q
    """
    assert (n - j) % 2 == 1, f"n-j must be odd, got n={n}, j={j}"

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


def shift_index(
    tensor: CartesianTensor | TensorProduct, shift: int
) -> CartesianTensor | TensorProduct:
    """
    Shift all the index of a tensor by a certain amount.

    For example, for T_ijk, and shift=1, the new tensor is T_ijl.

    Args:
        tensor: The tensor to shift the index.
        shift: The amount to shift the index.

    Returns:
        The new tensor with the shifted index.
    """

    def _shift(t: CartesianTensor):
        indices = "".join([chr(ord(i) + shift) for i in t.indices])
        return t.__class__(indices, factor=tensor.factor, symbol=t.symbol)

    if isinstance(tensor, CartesianTensor):
        return _shift(tensor)

    elif isinstance(tensor, TensorProduct):
        components = [_shift(t) for t in tensor]
        return tensor.__class__(*components, factor=tensor.factor)

    else:
        raise ValueError(f"Unknown tensor type: {type(tensor)}")


def shift_index_2(tensor: Tensors, shift: int) -> Tensors:
    """
    Shift all the index of a Tensors object by a certain amount.

    Returns:
        The new tensor with the shifted index.
    """
    components = [shift_index(t, shift) for t in tensor]
    return Tensors(*components)


def evaluate(
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


def evaluate_2(tensor: Tensors) -> Tensors:
    """Evaluate a linear combination of tensors."""
    components = [evaluate(t) for t in tensor]
    return Tensors(*components)


def contract_G(G1: Tensors, G2: Tensors, G1_indices: str, G2_indices: str) -> Tensors:
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

    return simplify_2(prod)


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


def canonize_delta_indices_2(tensor: Tensors) -> Tensors:
    components = [canonize_delta_indices(t) for t in tensor]
    return Tensors(*components)


def combine_terms(tensor: Tensors) -> Tensors:
    """
    Combine terms with the same indices. This currently only works for delta tensors.
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

    return Tensors(*all_combined)


def check_one(prod):
    evaluated = evaluate_2(prod)
    print("@@@ evaluated:", evaluated)

    canolized = canonize_delta_indices_2(evaluated)
    print("@@@ canolized:", canolized)

    combined = combine_terms(canolized)
    print("@@@ combined:", combined)


if __name__ == "__main__":
    all_G = G_odd(j=2, n=3)

    G1 = all_G[0]
    G2 = all_G[1]
    G3 = all_G[2]
    print("G_1:", G1)
    print("G_2:", G2)
    print("G_3:", G3)

    G2 = shift_index_2(all_G[1], shift=4)
    G3 = shift_index_2(all_G[2], shift=8)
    print("G_1, after shift:", G1)
    print("G_2, after shift:", G2)
    print("G_3, after shift:", G3)

    # check they are linearly independent
    prod = contract_G(G1, G2, "ABC", "EFG")
    print("=" * 40)
    check_one(prod)

    prod = contract_G(G1, G3, "ABC", "IJK")
    print("=" * 40)
    check_one(prod)

    prod = contract_G(G2, G3, "EFG", "IJK")
    print("=" * 40)
    check_one(prod)
