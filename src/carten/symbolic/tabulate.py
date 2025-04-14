"""
Tabulate tensors G, H, and S.

G, H, and S are made of only the Kronecker delta and Levi-Civita symbols.

G, H, and S can be used to map a general tensor T and a natural tensor X.
S = G H
X = H T
T' = G X = (G H) T = S T
where T' is the embedding of X in the T space.
"""

from fractions import Fraction
from pprint import pprint

import numpy as np
from monty.serialization import dumpfn

from carten.core.utils import dij, eijk, letter_index
from carten.symbolic.linearly_independent import (
    embed,
    get_G_even,
    get_g_matrix,
    get_G_odd,
    get_H,
    get_independent_H_coeff,
    get_K,
    group_G_by_symmetry,
    matrix_inverse,
    shift_index_2,
)
from carten.symbolic.symbolic_tensor import (
    Delta,
    Epsilon,
    LinearCombination,
    TensorProduct,
    multiply_2,
    simplify_2,
)
from carten.symbolic.utils import (
    find_independent_tensors,
    float_matrix,
    fraction_matrix,
)


def dij() -> np.ndarray:
    """Kronecker delta tensor.

    Returns:
        Kronecker delta tensor of shape (3, 3).
    """
    return np.eye(3)


def eijk():
    """Levi-Civita tensor.

    Returns:
        Levi-Civita tensor of shape (3, 3, 3).
    """
    e = np.zeros((3, 3, 3))
    e[0, 1, 2] = 1.0
    e[1, 2, 0] = 1.0
    e[2, 0, 1] = 1.0
    e[0, 2, 1] = -1.0
    e[1, 0, 2] = -1.0
    e[2, 1, 0] = -1.0

    return e


def tp_delta_epsilon(tp: TensorProduct, mode: str) -> np.ndarray:
    """Get the tensor product of Kronecker delta and Levi-Civita tensors.

    Note, the order of the indices need to be taken care of.
    Upper-case letters are used to represent tensors in the n space (namely for
    tensors T and such), while lower-case letters are used to represent tensors in the
    j space (namely for tensors X). So:
    1. X = H T: H would consist of both lower case and upper-case letters, and its
       upper-case letters are to be contracted with T. We assume the contracting rule
       is something like X_ab = H_abABC T_ABC.
    2. T' = G X: G would consist of both lower case and upper-case letters, and its
       lower-case letters are to be contracted with X. We assume the contracting rule is
       something like T'_ABC = G_ABCab X_ab.
    3. T' = G H T = S T: S would consist of only upper-case letters. We assume the
       contracting rule is something like T'_ABC = S_ABCDEF T_DEF, where the first
       half of the indices are associated with the embedded tensor T', while the
       latter half of the indices are associated with the original tensor T.

    Args:
        tp: Tensor product of Kronecker delta and Levi-Civita tensors.
        mode: which mode to use, either `G`, `H`, or `S`. This determines how the
            output indices are ordered.

    Returns:
        Tensor product of Kronecker delta and Levi-Civita tensors.
    """
    delta_rules = []
    epsilon_rules = []
    for t in tp.components:
        if isinstance(t, Delta):
            delta_rules.append(t.indices)
        elif isinstance(t, Epsilon):
            epsilon_rules.append(t.indices)
        else:
            raise ValueError(f"Unknown tensor type: {type(t)}")

    left = ",".join(delta_rules + epsilon_rules)

    # Since the tensors only consists of delta and epsilon, the left rule should be OK,
    # but the right rule should be ordered according to the mode.
    right = "".join(delta_rules + epsilon_rules)
    lower = sorted([c for c in right if c.islower()])
    upper = sorted([c for c in right if c.isupper()])
    if mode == "G" or mode == "S":
        right = "".join(upper + lower)
    elif mode == "H":
        right = "".join(lower + upper)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    rule = left + "->" + right

    d = dij()
    e = eijk()
    deltas = [d for _ in range(len(delta_rules))]
    epsilons = [e for _ in range(len(epsilon_rules))]
    data = deltas + epsilons

    product = np.einsum(rule, *data)

    # multiply factor
    product = product * tp.factor

    return product


def evaluate_tensors(tensors: LinearCombination, mode: str) -> np.ndarray:

    # Evaluate each tensor product
    output = []
    for tp in tensors.components:
        if isinstance(tp, TensorProduct):
            v = tp_delta_epsilon(tp, mode)
            output.append(v)
        else:
            raise ValueError(f"Unknown tensor type: {type(tp)}")

    # Sum over all tensor products
    sum = np.sum(output, axis=0)

    return sum.tolist()


def get_G_H_S_of_j(j: int, n: int, symmetry: str = None) -> tuple[
    list[LinearCombination],
    list[LinearCombination],
    list[LinearCombination],
    list[list[Fraction]],
    list[list[Fraction]],
]:
    """
    Get the G_H_S tensors for a given weight j and dimension n.

    Args:
        j: weight
        n: dim of the space T is in
        symmetry:

    Returns:
        G: independent G tensors of different seniority p
        H: H corresponding to G
        S: S corresponding to G and H
        g: g_pq matrix
        h: g_pq matrix
    """
    # create G mapping operator
    if (n - j) % 2 == 0:
        all_G = get_G_even(j, n)
    else:
        all_G = get_G_odd(j, n)

    # Get numerical S tensors, embedding space j to space n
    all_num_S = [embed(j, G) for G in all_G]

    # Get linearly independent S tensors
    _, independent_indices = find_independent_tensors(all_num_S)

    # Get linearly independent G tensors
    independent_G = [all_G[i] for i in independent_indices]

    # Get g_pq matrix for independent G
    g_pq = get_g_matrix(j, n, independent_G)

    # Get h_pq matrix
    h_pq = matrix_inverse(g_pq)

    # Get H tensors, corresponding to independent G
    independent_H = get_H(h_pq, independent_G)

    # Further down select unique G tensors by symmetry
    if symmetry is not None:
        indices_zero, indices_group = group_G_by_symmetry(independent_G, n, symmetry)

        # All G result in zero
        if len(indices_group) == 0:
            all_K = []
        # Each G form its own group, i.e. all G are independent
        elif len(indices_group) == len(independent_G):
            all_K = independent_G
        # Some G are not unique
        else:
            # TODO, These two can be combined as a single function
            coeff = get_independent_H_coeff(h_pq, indices_group)
            all_K = get_K(independent_G, coeff, indices_group)

        # We use K as G now
        independent_G = all_K
        independent_H = independent_H[: len(all_K)]

    # Get G, H, and S tensors
    all_G = []
    all_H = []
    all_S = []
    for i, (G, H) in enumerate(zip(independent_G, independent_H)):
        G = simplify_2(G)

        # Shift upper letters of H to distinguish those from G
        H = shift_index_2(H, n, letter_index(24, upper_case=True))
        H = simplify_2(H)

        S = multiply_2(G, H)
        S = simplify_2(S)

        all_G.append(G)
        all_H.append(H)
        all_S.append(S)

    return all_G, all_H, all_S, g_pq, h_pq


def get_G_H_S(n: int, symmetry: str = None, numerical: bool = True) -> dict:
    """
    Get all the G, H, S tensors of dimension n.

    Args:
        n: dim of the space T is in
        symmetry: symmetry of the tensor in space n, if any.
        numerical: whether to return numerical values of G, H, S.

    Returns:
        G, H, S, and g_pq, h_pq information.
    """
    out = {}
    for j in range(n + 1):
        G, H, S, g, h = get_G_H_S_of_j(j, n, symmetry)
        out_j = {
            "g_pq": {"symbolic": fraction_matrix(g), "numerical": float_matrix(g)},
            "h_pq": {"symbolic": fraction_matrix(h), "numerical": float_matrix(h)},
            "G": [],
            "H": [],
            "S": [],
        }

        # loop over seniority p
        for G_p, H_p, S_p in zip(G, H, S):

            lower = letter_index(j)
            upper = letter_index(n, upper_case=True)
            upper2 = letter_index(n, start=n, upper_case=True)

            # G
            out_j["G"].append(
                {"symbolic": str(G_p), "rule": (f"{upper}{lower},{lower}->{upper}")},
            )
            if numerical:
                out_j["G"][-1]["numerical"] = evaluate_tensors(G_p, mode="G")

            # H
            out_j["H"].append(
                {"symbolic": str(H_p), "rule": f"{lower}{upper},{upper}->{lower}"}
            )
            if numerical:
                out_j["H"][-1]["numerical"] = evaluate_tensors(H_p, mode="H")

            # S
            out_j["S"].append(
                {"symbolic": str(S_p), "rule": f"{upper}{upper2},{upper2}->{upper}"}
            )
            if numerical:
                out_j["S"][-1]["numerical"] = evaluate_tensors(S_p, mode="S")

        out[j] = out_j

    return out


if __name__ == "__main__":

    # # elastic tensor
    # j = 4
    # rank = 4
    # symmetry = "ijkl=jikl=klij"
    # get_G_H_S_of_j(j, rank, symmetry)

    ######
    rank = 4
    symmetry = "ijkl=jikl=klij"
    out = get_G_H_S(rank, symmetry, numerical=False)
    pprint(out)
    # dumpfn(out, "out.yaml")
