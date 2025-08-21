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
    L1: int, L2: int, L3: list[int], mode: str = "full", level: int = None
) -> dict[int, list[tuple[int, int, int]]]:
    """Get the paths from L1 and L2 to L3.


    Args:
        L1: maximum rank of the natural tensor in the first input feature tensor.
        L2: maximum rank of the natural tensor in the second input feature tensor.
        L3: ranks of the output feature tensor.
        mode: method to select the paths. Supported modes are `full`, `camp`,
            `lite`, and `level`.

            In a nutshell:
            - `full` allows all paths, but can be computationally expensive.
            - `camp` and `lite` are designed to allow message passing from higher-rank
               tensors to lower-rank tensors. This is good for modeling low-rank values,
               e.g. the scalar interatomic potential energy. One is suggested to use
               `lite` instead of `camp` for symmetric behaviors (see below).
            - `level` favors interactions of between low-rank tensors and balances
               tensors of all ranks. It provides a way to limit the number of paths,
               and it is fully controllable. For good choice of `level`, see below.

            For all modes discussed below, it is required that:
            - abs(l1 - l2) <= l3 <= l1 + l2;
            - l3 should be in L3.
            And additional rules apply for each mode.

            Z = X @ Y, where l1 is the rank of X, l2 is the rank of Y, and l3 is the
            rank of Z.

            In `full` mode, there is no additional restriction on the paths.

            In `camp` mode, the paths are selected in the same way as in the CAMP model.
            It requires l1 to be fully contracted, and as a result,
            l3 = l1 + l2 - 2 * l1.

            The camp mode is not symmetric, meaning that the order of the two tensors
            matters. Z = X @ Y, where X is fully contracted, and Y is not. This can
            be non-optimal since X is typically the features and Y the dyadics.

            In the `lite` mode, we make the `camp` mode symmetric, allowing:
            l3 = l1 + l2 - 2 * l1 and l3 = l1 + l2 - 2 * l2.

            If mode == `level`, additional argument `level` (see below) is used to
            specify the maximum allowed sum of l1 and l2. The paths are selected
            based on:
            l1+l2 <= level.

            Note, this is not applied for l3=0 (scalars), but only for higher ranks.

            This becomes the same as the `full` mode when level is set to a value larger
            than `L1 + L2`. A good choice for `level` to start with is max(L1, L2).

        level: level value for the `level` mode. Ignored for other modes.

    Returns:
        Dictionary of paths from L1 and L2 to L3: {l3: [(l1, l2, l3)]}, where each
        tuple is a valid path from l1 and l2 to l3.
    """
    paths = defaultdict(list)

    if level is not None and mode != "level":
        raise ValueError(
            "`level` is provided,  but not needed for mode other than `level`."
            "Set to `None` if not needed."
        )

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
                else:
                    l = l1 + l2 - 2 * l1
                if l in L3:
                    paths[l].append((l1, l2, l))
            elif mode == "level":
                if level is None:
                    raise ValueError("level must be specified when mode is 'level'.")
                for l in range(abs(l1 - l2), l1 + l2 + 1):
                    # For scalars, we don't apply the level restriction
                    if l != 0 and l1 + l2 > level:
                        continue
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
