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
        mode: how to compute the paths. Supported modes are `full`, `mid`, `lite`.
            In `full` mode, all paths satisfying `abs(l1 - l2) <= l3 <= l1 + l2` are
            generated.
            In `mid` mode, the upper bound is constrained to `max(l1, l2)`, meaning
            paths satisfying `abs(l1 - l2) <= l3 <= max(l1, l2)` are generated. By
            doing this, information are allowed to flow from high-rank tensors to
            low-rank tensors, but not the other way around.
            # TODO, the description of lite is incorrect.
            In `lite` mode, the upper bound is set to `abs(l1-l2)`, the same as lower.
            The is going to be exactly the same rule as in `CAMP`. The idea is
            that `tensor contraction` (instead of tensor product), is used.
            For example, given l1=2, l2=2: `full` mode will give l3 in [0, 1, 2, 3, 4];
            `mid` mode will give l3 in [0, 1, 2], not exceeding the ranks of the input
           tensors; `lite` mode will only allow l3 in [0, 2], corresponding to
           contraction once or twice of the indices of input tensors.

    Returns:
        Dictionary of paths from L1 and L2 to L3: {l3: [(l1, l2, l3)]}, where each
        tuple is a valid path from l1 and l2 to l3.
    """
    paths = defaultdict(list)

    for l1 in range(L1 + 1):
        for l2 in range(L2 + 1):
            if mode in ["full", "mid"]:
                if mode == "full":
                    upper = l1 + l2 + 1
                else:
                    upper = max(l1, l2) + 1

                for l in range(abs(l1 - l2), upper):
                    if l in L3:
                        paths[l].append((l1, l2, l))
            # TODO, modify the docs above, this is not the same as in CAMP, where the
            #  lower-rank tensors are fully contracted away, meaning l1=2 and l2=4
            #  will only lead to l3=2, (contracting both indices of l1); and it is
            #  impossible to have l3=4 (contracting only one index of l1).
            elif mode == "lite":
                for i in range(1, (l1 + l2) // 2 + 1):
                    if i > l1 or i > l2:
                        continue
                    l = l1 + l2 - 2 * i
                    if l in L3:
                        paths[l].append((l1, l2, l))
            else:
                raise ValueError(f"Invalid mode: {mode}.")

    return paths


if __name__ == "__main__":
    paths1 = get_paths(4, 4, [0, 1, 2, 3, 4], mode="full")
    paths2 = get_paths(4, 4, [0, 1, 2, 3, 4], mode="mid")
    paths3 = get_paths(4, 4, [0, 1, 2, 3, 4], mode="lite")
    pprint(paths1)
    pprint(paths2)
    pprint(paths3)
