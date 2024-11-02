"""Find the unique natural tensors in a reduction spetrum.

For example, for a rank-5 tensor T_ijklm, the rank-2 natural tensors can be generated
from 30 components:
- S_am = T_ijklm \delta_ij \epsilon_akl
- S_am = T_ijkml \delta_ij \epsilon_akl
- S_am = T_ijmkl \delta_ij \epsilon_akl

- S_am = T_ikjlm \delta_ij \epsilon_akl
- S_am = T_ikjml \delta_ij \epsilon_akl
- S_am = T_imjkl \delta_ij \epsilon_akl
...
There are 30 components in total because C(5, 2) = 10 options to choose delta_ij, and
C(3, 2) = 3 options to choose k, l.

However, we know from the reduction spectrum that only 15 components are unique.
Here, we try to find the unique components.
"""
from IPython.core.release import kernel_protocol_version_info

from carten.utils import letter_index
import itertools


def get_candidates(rank: int) -> list[str]:
    """
    Get all the candidate tensors S for a given rank.

    The candidates are obtained by choosing two slots to place the indices `i` and `j`
    (they are contracted with delta_ij) and two slots to place the indices `k`, `l`
    (they are contracted with the Levi-Civita tensor) in all the possible ways.

    Args:
        rank: the rank of the tensor T
    """
    remaining = letter_index(rank - 4, start=ord("m") - ord("a"))

    candidates = []
    for i_pos, j_pos in itertools.combinations(range(rank), 2):
        untaken1 = sorted(set(range(rank)) - {i_pos, j_pos})
        for k_pos, l_pos in itertools.combinations(untaken1, 2):
            untaken2 = sorted(set(untaken1) - {k_pos, l_pos})
            cand = [None for _ in range(rank)]
            cand[i_pos] = "i"
            cand[j_pos] = "j"
            cand[k_pos] = "k"
            cand[l_pos] = "l"
            for m, pos in zip(remaining, untaken2):
                cand[pos] = m

            candidates.append("".join(cand))

    return candidates


def symmetrize(candidate: str) -> dict[str, dict[str, str]]:
    """
    Symmetrize a candidate string S by permuting the indices.

    For example, for a candidate T_ijklm, its corresponding S tensor is
    S_hm = \delta_ij \epsilon_hkl T_ijklm.

    Here we permute the indices in S that can symmetrize the tensor S.

    Returns:
        {S: {delta:s1, eps:s1, T:s2}} dictionary, where S is a symmetrized tensor
            version of S_hm, and s1, s2, and s3 are the corresponding permutations of
            delta, epsilon and T that makes S.

    """

    # positions to place indices other than i, j, k, l
    remaining = [c for c in candidate if c not in "ijkl"]
    remaining_idx = [i for i, c in enumerate(candidate) if c not in "ijkl"]

    indices = ["h"] + remaining

    out = {}
    for perm in itertools.permutations(indices):
        perm = "".join(perm)

        # first used for epsilon
        eps = perm[0] + "kl"

        # the others are used in T
        cand = list(candidate)
        for i, s in zip(remaining_idx, perm[1:]):
            cand[i] = s

        out[perm] = {"delta": "ij", "eps": eps, "T": "".join(cand)}

    return out


def delta_epsilon_T(T: str, mapping: {str: int}) -> list[str]:
    """
    Evaluate a tensor:
    S_hklm... = delta_ij \epsilon_hkl T_\pi{ijklm...},
    where \pi{ijkl...} are some permutations of the indices.

    Note, \epsilon_hkl is the Levi-Civita tensor.

    Args:
        T: a string representing the tensor T.
        mapping: a dictionary that maps the indices h, k, l, m, ... to integers, 1,
            2 or 3. No mapping is needed for i and j, it will be automatically
            determined from the mapping of h. If an index is not in the mapping or
            if the value is None, no substitution is made for that index.

    Returns:
       A list of six strings of T, where the first three strings are the positive term
       and the next three are the negative term from the Levi-Civita tensor.
       The each group of three are associated with delta_ij.
    """
    keys = set(mapping.keys())
    assert "h" in keys, "h should be in the mapping"

    for idx in ["i", "j", "k", "l"]:
        assert idx not in keys, f"{idx} should not be in the mapping"

    keys.remove("h")
    assert keys.issubset(set(T)), "All the keys should be in the tensor T"

    # Remove indices with None values
    mapping = {k: str(v) for k, v in mapping.items() if v is not None}

    h = mapping["h"]

    if h == "1":
        mapping["k"] = "2"
        mapping["l"] = "3"
    elif h == "2":
        mapping["k"] = "3"
        mapping["l"] = "1"
    elif h == "3":
        mapping["k"] = "1"
        mapping["l"] = "2"
    else:
        raise ValueError("h should be 1, 2, or 3")

    # delta ij, result in a summation over i, j, taking values 1, 2, 3
    candidates = [T.replace("i", str(x)).replace("j", str(x)) for x in [1, 2, 3]]

    out = []

    # epsilon_hkl
    for cand in candidates:
        for k, v in mapping.items():
            cand = cand.replace(k, v)
        out.append(cand)

    # epsilon_hlk
    # flip k and j, and sign
    mapping["k"], mapping["l"] = mapping["l"], mapping["k"]
    for cand in candidates:
        for k, v in mapping.items():
            cand = cand.replace(k, v)
        out.append("-" + cand)

    return out


def check_linear_dependence(
    candidates: list[str], mapping: dict[str, int]
) -> tuple[bool, list[int], dict]:
    """
    Check if there are linear dependencies among the candidate tensors.

    For all the checking, we check the ones associated with U_hmn, which consists
    of S_hmn and S_hnm, i.e., the ones starting with h.

    Args:
        candidates: list of candidate tensors
        mapping: a dictionary that maps the indices h, k, l, m, ... to integers, 1,
            2 or 3.
            No mapping is needed for i and j, since it will result in a summation,
            over 1,2,3.
            No mapping is needed for k and j, it will be automatically
            determined from the mapping of h, since they are associated with epsilon.
            If an index is not in the mapping or
            if the value is None, no substitution is made for that index.
    """

    out = {}  # record all info

    data_for_check = []
    for cand in candidates:
        U = symmetrize(cand)

        # select the ones starting with h
        U = {k: v for k, v in U.items() if k[0] == "h"}

        out[cand] = {}
        out[cand]["U"] = U
        out[cand]["delta_epsilon_T"] = {}

        # evaluate delta_ij and epsilon_hkl T...
        cand_data = []
        for k, v in U.items():
            del_eps_T = delta_epsilon_T(v["T"], mapping=mapping)
            out[cand]["delta_epsilon_T"][k] = del_eps_T
            cand_data.extend(del_eps_T)

        data_for_check.append(cand_data)

    found, signs = is_dependent(data_for_check)

    return found, signs, out


def is_dependent(candidates: list[list[str]]) -> tuple[bool, tuple[int, ...]]:
    """
    Check if a list of list of strings are linearly dependent.

    For example,
    [[a, b], [-b, c], [a, c]] is linearly dependent because
    if [a, b]  + [-b, c] - [a, c] = 0

    Args:
        candidates:

    Returns:
        found: True if there is a linear dependence
        found_sign: the sign of the linear dependence
    """
    n = len(candidates)
    signs = itertools.product([-1, 1], repeat=n)

    found = False
    found_sign = None

    # loop over all combinations of signs
    for sign in signs:
        # we put all tensors into two groups based on their sign, and
        # check if the two groups are the same without considering the sign
        positive = []
        negative = []

        # loop over all the candidates and the corresponding sign
        for s, cand in zip(sign, candidates):
            # loop over all the tensors in a candidate
            for t in cand:
                # four cases
                if s == 1 and "-" not in t:
                    positive.append(t)
                elif s == 1 and "-" in t:
                    negative.append(t)
                elif s == -1 and "-" not in t:
                    negative.append("-" + t)
                elif s == -1 and "-" in t:
                    positive.append(t[1:])
                else:
                    raise ValueError(f"Invalid combinations s{s}, t{t}")

        # remove the negative sign and compare
        negative = [t.replace("-", "") for t in negative]
        if sorted(positive) == sorted(negative):
            found = True
            found_sign = sign
            break

    return found, found_sign


def check_one(to_check, mapping, indices, only_print_dependent: bool = True):
    dependent, signs, info = check_linear_dependence(to_check, mapping=mapping)

    if (not only_print_dependent) or dependent:
        print("\n\n" + "=" * 80)
        print(indices)

        print("\n\n" + "=" * 40)
        print("Dependent:", dependent)
        print("Signs:", signs)
        # print(info)

        for i, cand in enumerate(to_check):
            with_mapping = info[cand]["delta_epsilon_T"]
            print("Candidate:", cand)
            print("  Values after mapping:", with_mapping)


if __name__ == "__main__":
    # ##
    # print("\n\n" + "=" * 40)
    # rank = 5
    # S = get_candidates(rank)
    # print("All candidates:")
    # for i, s in enumerate(S):
    #     print(i, s)
    #
    # for s in S:
    #     print("\n\n" + "=" * 40)
    #     print("Candidate:", s)
    #     U = symmetrize(s)
    #
    #     s0 = None
    #     for s, v in U.items():
    #         # print empty line if a new group starts
    #         if s0 is None:
    #             print()
    #         else:
    #             if s[0] != s0:
    #                 print()
    #         s0 = s[0]
    #         print(f"S_{s},  delta_{v['delta']}, eps_{v['eps']},  T_{v['T']}")

    ###
    ### example
    rank = 5
    S = get_candidates(rank)
    print("All candidates:")

    for i, s in enumerate(S):
        print(i, s)

    # if not setting the mapping, the values will not be substituted
    mapping = {
        "h": 1,
        # "i": 2,
        # "j": 2,
        # "k": 2,
        # "l": 3,
        "m": 2,
    }

    for comb in itertools.combinations(range(len(S)), r=3):
        to_check = [S[i] for i in comb]
        check_one(to_check, mapping, list(comb), only_print_dependent=True)

    print("Done!")
