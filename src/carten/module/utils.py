from collections import defaultdict


def check_rank(L1: int, L2: int, L3: int | tuple[int, ...] | None) -> tuple[int, ...]:
    """Helper function to get valid L3."""

    if isinstance(L3, int):
        L3 = range(abs(L1 - L2), L3 + 1)
    elif isinstance(L3, (tuple, list)):
        allowed = set(range(abs(L1 - L2), max(L1, L2) + 1))
        if not set(L3).issubset(allowed):
            raise ValueError(f"Invalid L3: {L3}. Allowed values are {allowed}.")
        L3 = sorted(L3)
    elif L3 is None:
        L3 = range(abs(L1 - L2), max(L1, L2) + 1)
    else:
        raise ValueError(f"Invalid L3: {L3}. Must be int, tuple, list, or None.")

    return tuple(sorted(L3))


def get_paths(
    L1: int, L2: int, L3: tuple[int, ...]
) -> dict[int, list[tuple[int, int, int]]]:
    """Get the paths from L1 and L2 to L3.

    Args:
        L1: maximum rank of the natural tensor in the first input feature tensor.
        L2: maximum rank of the natural tensor in the second input feature tensor.
        L3: ranks of the output feature tensor.

    Returns:
        Dictionary of paths from L1 and L2 to L3: {l3: [(l1, l2, l3)]}, where each
        tuple is a valid path from l1 and l2 to l3.
    """
    paths = defaultdict(list)

    for l1 in range(L1 + 1):
        for l2 in range(L2 + 1):
            for l in range(abs(l1 - l2), l1 + l2 + 1):
                if l in L3:
                    paths[l].append((l1, l2, l))

    return paths
