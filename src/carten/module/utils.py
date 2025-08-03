from collections import defaultdict
from pprint import pprint


def check_rank(L1: int, L2: int, L3: int | list[int] | None) -> list[int]:
    """Helper function to get valid l3.

    Convert all values to a tuple of l3 for the give L1, L2, and L3.
    """

    if isinstance(L3, int):
        if not 0 <= L3 <= L1 + L2:
            raise ValueError(f"Invalid L3: {L3}. Must be in [0, L1 + L2].")
        L3 = range(L3 + 1)
    elif isinstance(L3, (tuple, list)):
        allowed = set(range(L1 + L2 + 1))
        if not set(L3).issubset(allowed):
            raise ValueError(
                f"Invalid L3: {L3}. For L1={L1} and L2={L2}, allowed values are "
                f"{allowed}."
            )
    elif L3 is None:
        L3 = range(max(L1, L2) + 1)
    else:
        raise ValueError(f"Invalid L3: {L3}. Must be int, tuple, list, or None.")

    return sorted(L3)


def get_paths(
    L1: int, L2: int, L3: list[int], mode: str = "full"
) -> dict[int, list[tuple[int, int, int]]]:
    """Get the paths from L1 and L2 to L3.

    Args:
        L1: maximum rank of the natural tensor in the first input feature tensor.
        L2: maximum rank of the natural tensor in the second input feature tensor.
        L3: ranks of the output feature tensor.
        mode: how to compute the paths. Supported modes are `full`, `camp`, and `lite`.
            In `full` mode, all paths satisfying `abs(l1 - l2) <= l3 <= l1 + l2` are
            generated.
            The `camp` is the same rule as used in the CAMP model. See the supplemental
            information of the CAMP paper for details.
            The `lite` model is the same as `camp`, but with switching the order
            of the two tensors. In the `camp` mode, give two tensors X and Y,
            we require the rank of X is smaller than or equal to the rank of Y,
            and requiring that X is fully contracted, Here, we do the reverse, requiring
            that the rank of Y is smaller than or equal to the rank of X, and requiring
            that Y is fully contracted. Why? Because in the implementation, X is
            typically the features and Y is typically the dyadics. Then in the `camp`
            mode, important features can be discarded. This may not be a huge problem
            for interatomic potentials like CAMP, but can be bad for modeling high-rank
            tensors.
            In `lite` mode, the tensors X of rank l1 and Y of rank l2 are to be
            contracted by rank k. Same as the `camp` mode, it is required that
            l3 = l1 + l2 - 2 * k, but unlike `camp`, where it is required that
            l1<l2, here, we do not require that. Also, `camp` requires l1 to be fully
            contracted, while `lite` does not.
            In terms of the number of allowed paths, full>lite>camp.

    Returns:
        Dictionary of paths from L1 and L2 to L3: {l3: [(l1, l2, l3)]}, where each
        tuple is a valid path from l1 and l2 to l3.
    """
    paths = defaultdict(list)

    for l1 in range(L1 + 1):
        for l2 in range(L2 + 1):
            if mode == "full":
                for l in range(abs(l1 - l2), l1 + l2 + 1):
                    if l in L3:
                        paths[l].append((l1, l2, l))
            elif mode == "camp":
                if l1 <= l2:
                    l = l1 + l2 - 2 * l1
                    if l in L3:
                        paths[l].append((l1, l2, l))
            elif mode == "lite":
                if l2 <= l1:
                    l = l1 + l2 - 2 * l2
                    if l in L3:
                        paths[l].append((l1, l2, l))
            else:
                raise ValueError(f"Invalid mode: {mode}.")

    return paths


if __name__ == "__main__":

    L1 = 3
    L2 = 3
    L3 = list(range(3 + 1))
    for mode in ["full", "lite", "camp"]:
        paths = get_paths(L1, L2, L3, mode)
        print("\n" + "#" * 40)
        print(f"mode: {mode}")
        print({k: len(v) for k, v in paths.items()})
        pprint(paths)
