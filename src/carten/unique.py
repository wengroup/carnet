"""Find the unique natural tensors in a reduction spetrum.

For example, for a rank-4 tensor T_ijkl, the rank-3 natural tensors can be generated
from the below 6 tensors S_ijk =:
- \epsilon_{hij} T_{ijkl}
- \epsilon_{hij} T_{ikjl}
- \epsilon_{hij} T_{iklj}
- \epsilon_{hij} T_{kijl}
- \epsilon_{hij} T_{kilj}
- \epsilon_{hij} T_{klij}

However, we know from the reduction spectrum of a general rank-4 tensor:
4 = (0)*3 + (1)*6 + (2)*6 + (3)*3 + (4)*1,
meaning that the reduction spectrum consists of:
- 3 rank-0 natural
- 6 rank-1 natural
- 6 rank-2 natural
- 3 rank-3 natural
- 1 rank-4 natural

Therefore, the above 6 rank-3 natural tensors are not unique, and we need to find the
"""

import itertools

from carten.utils import letter_index


def get_candidates(rank: int) -> list[str]:
    """
    Get all the candidate tensors S for a given rank.

    The candidates are obtained by choosing two slots to place the indices `i` and `j`
    (they are contracted with the Levi-Civita tensor) in all the possible ways.
    For example, for a rank-4 tensor T_ijkl, the candidate tensors S_ijk are:
    - T_{ijkl}
    - T_{ikjl}
    - T_{iklj}
    - T_{kijl}
    - T_{kilj}
    - T_{klij}

    Args:
        rank: the rank of the tensor T
    """
    remaining = letter_index(rank - 2, start=ord("k") - ord("a"))

    candidates = []
    for i_pos, j_pos in itertools.combinations(range(rank), 2):
        cand = list(remaining)
        cand.insert(i_pos, "i")
        cand.insert(j_pos, "j")
        candidates.append("".join(cand))

    return candidates


def symmetrize(candidate: str) -> dict[str, dict[str, str]]:
    """
    Symmetrize a candidate string S by permuting the indices.

    For example, for a candidate T_ijkl, its corresponding S tensor is
    S_hkl = \epsilon_hij T_ijkl.

    Here we permute the indices in S that can symmetrize the tensor S.

    Returns:
        {S: {eps:s1, T:s2}} dictionary, where S is a symmetrized tensor version of
        S_hkl, and s1 and s2 are the corresponding permutations of epsilons and T that
        makes S.

    """

    # Find the positions of the indices i and j
    i_pos = candidate.index("i")
    j_pos = candidate.index("j")

    # Indices of S to be permuted
    indices = "h" + candidate.replace("i", "").replace("j", "")

    out = {}
    for p in itertools.permutations(indices):
        p = "".join(p)

        eps = p[0] + "ij"

        T = list(p[1:])
        T.insert(i_pos, "i")
        T.insert(j_pos, "j")

        out[p] = {"eps": eps, "T": "".join(T)}

    return out


def epsilon_T(T: str, mapping: {str: int}) -> tuple[str, str]:
    """
    Evaluate a tensor:
    S_hklm... = \epsilon_hij T_\pi{ijklm...},
    where \pi{ijkl...} are some permutations of the indices.

    Note, \epsilon_hij is the Levi-Civita tensor.

    Args:
        T: a string representing the tensor T.
        mapping: a dictionary that maps the indices h, k, l, m, ... to integers, 1,
            2 or 3. No mapping is needed for i and j, it will be automatically
            determined from the mapping of h. If an index is not in the mapping or
            if the value is None, no substitution is made for that index.

    Returns:
       A tuple of two strings, where the first string is the positive term and the
         second string is the negative term from the Levi-Civita tensor.
    """
    keys = set(mapping.keys())
    assert "h" in keys, "h should be in the mapping"
    keys.remove("h")
    assert keys.issubset(set(T)), "All the keys should be in the tensor T"

    # Remove indices with None values
    mapping = {k: str(v) for k, v in mapping.items() if v is not None}

    h = mapping["h"]

    if h == "1":
        mapping["i"] = "2"
        mapping["j"] = "3"
    elif h == "2":
        mapping["i"] = "3"
        mapping["j"] = "1"
    elif h == "3":
        mapping["i"] = "1"
        mapping["j"] = "2"
    else:
        raise ValueError("h should be 1, 2, or 3")

    # epsilon_hij
    out1 = T
    for k, v in mapping.items():
        out1 = out1.replace(k, v)

    # epsilon_hji
    # flip i and j, and sign
    mapping["i"], mapping["j"] = mapping["j"], mapping["i"]
    out2 = T
    for k, v in mapping.items():
        out2 = out2.replace(k, v)
    out2 = "-" + out2

    return [out1, out2]


def check_linear_dependence(
    candidates: list[str], mapping: dict[str, int]
) -> tuple[bool, list[int], dict]:
    """
    Check if there are linear dependencies among the candidate tensors.

    For all the checking, we check the ones associated with U_hkl, which consits
    of S_hkl and S_hkl, i.e., the ones starting with h.

    Args:
        candidates: list of candidate tensors
        mapping: a dictionary that maps the indices h, k, l, m, ... to integers, 1,
            2 or 3. No mapping is needed for i and j, it will be automatically
            determined from the mapping of h. If an index is not in the mapping or
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
        out[cand]["epsilon_T"] = {}

        cand_data = []
        for k, v in U.items():
            edT = epsilon_T(v["T"], mapping=mapping)
            out[cand]["epsilon_T"][k] = edT
            cand_data.extend(edT)

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


def check_one(to_check, mapping, only_print_dependent: bool = True):
    dependent, signs, info = check_linear_dependence(to_check, mapping=mapping)

    if (not only_print_dependent) or dependent:
        print("\n\n" + "=" * 40)
        print("Dependent:", dependent)
        print("Signs:", signs)
        # print(info)

        for i, cand in enumerate(to_check):
            with_mapping = info[cand]["epsilon_T"]
            print("Candidate:", cand)
            print("  Values after mapping:", with_mapping)


if __name__ == "__main__":
    # print("\n\n" + "=" * 40)
    # print(f"Rank {rank}, number of candidates: {len(S)}")
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
    #
    #         print(f"S_{s},  eps_{v['eps']},  T_{v['T']}")
    #

    ###
    ### example
    rank = 4
    S = get_candidates(rank)
    print("All candidates:")
    for i, s in enumerate(S):
        print(i, s)

    # if not setting the mapping, the values will not be substituted
    mapping = {
        "h": 1,
        "k": 2,
        "l": 3,
        # "m": 2,
    }

    for comb in itertools.combinations(range(len(S)), r=3):
        print("\n\n" + "=" * 80)

        print(list(comb))
        to_check = [S[i] for i in comb]
        check_one(to_check, mapping, only_print_dependent=True)
