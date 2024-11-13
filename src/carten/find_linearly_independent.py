"""
Find linearly independent natural tensors.

Reference:

References:
    Eq 19, Eq 21, Table 1, Irreducible Cartesian Tensors. II. General Formulation,
    http://dx.doi.org/10.1063/1.1665190

"""
import itertools
from fractions import Fraction

from carten.reduce import get_permutations_2
from carten.symbolic_tensor import Delta, TensorProduct, Tensors
from carten.utils import letter_index


def E(j: int) -> Tensors:
    """
    Projectors to get natural tensors of rank j: E(j | j).

    References:
        Eq 19, Eq 21, Table 1, Irreducible Cartesian Tensors. II. General Formulation,
        http://dx.doi.org/10.1063/1.1665190

    Args:
        j: rank of the projection operator

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

        # converting dict rules to list rules
        all_rules = expand_delta_rules(j, t)
        list_rules = [rule["d_rs"] + rule["d_rr"] + rule["d_ss"] for rule in all_rules]

        # create tensor products of deltas for each rule
        deltas = create_delta_tensors(list_rules, factor=c)
        out.extend(deltas)

        print(f"@@@ debug E_j: j={j}, t={t}, c={c}")

    return Tensors(*out)


def create_delta_tensors(
    rules: list[list[str]], factor: int | Fraction = 1
) -> list[TensorProduct]:
    """Create a Tensors object for deltas given the rules.

    Args:
        rules: Each inner list contains the indices for the deltas.
        factor: additional factor to multiply with the tensor product

    Returns:
        List of TensorProduct objects.
    """
    # final factor is equal to 1 / len(rules) multiple the given factor
    factor = Fraction(1, len(rules)) * factor

    all_tp = []
    for rule in rules:
        tensors = [Delta(pair) for pair in rule]
        tp = TensorProduct(*tensors, factor=factor)
        all_tp.append(tp)

    return all_tp


def expand_delta_rules(j: int, t: int) -> list[dict[str : list[str]]]:
    """
    Rules for d_{rs}^{j-2t} d_{rr}^t d_{ss}^t.

    This is in Eq. 19 of the paper.

    Args:
        j: rank of the projection operator
        t: number of d_rr and d_ss

    Examples:
        >>> expand_delta_rules(3, 1)
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
