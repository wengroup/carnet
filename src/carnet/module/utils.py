from collections import defaultdict
from pprint import pprint

from torch import Tensor, nn


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
    L1: int,
    L2: int,
    L3: list[int],
    mode: str = "full",
    level: int = None,
    polar_only: bool = False,
    l2_even_parity: bool = False,
    downward: bool = False,
) -> dict[int, list[tuple[int, int, int]]]:
    """Get the paths from L1 and L2 to L3.


    Args:
        L1: maximum rank of the natural tensor in the first input feature tensor.
        L2: maximum rank of the natural tensor in the second input feature tensor.
        L3: ranks of the output feature tensor.

        mode: method to select the paths. Supported modes are `full`, `camp`,
            `lite`, and `level`, and `level2`.

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

            If mode == `level`, additional argument `level` (see below) is used to
            specify the maximum allowed sum of l1 and l2. The paths are selected
            based on:
            l1+l2 <= level.
            Note, this is not applied for l3=0 (scalars), but only for higher ranks.
            This becomes the same as the `full` mode when level is set to a value larger
            than `L1 + L2`. A good choice for `level` to start with is max(L1, L2).

            For the `level2` mode, everything is the same as `level` mode, except that
            the restriction is:
            l1 + l2 + l3 <= level.
            This is similar to the `MTP` way of selecting the basis functions.

            In `camp` mode, the paths are selected in the same way as in the CAMP model.
            It requires l1 to be fully contracted, and as a result,
            l3 = l1 + l2 - 2 * l1.

            The camp mode is not symmetric, meaning that the order of the two tensors
            matters. Z = X @ Y, where X is fully contracted, and Y is not. This can
            be non-optimal since X is typically the features and Y the dyadics.

            In the `lite` mode, we make the `camp` mode symmetric, allowing:
            l3 = l1 + l2 - 2 * l1 or l3 = l1 + l2 - 2 * l2.


        level: level value for the `level` mode. Ignored for other modes.

        polar_only: If `Ture`, only include paths that produce polar tensors. This
            should be used together with `l2_even_parity` (see below).

        l2_even_parity: Whether the tensor associated with l2 is always of even
            time reversal symmetry. This should be used together with `polar_only`.

            `polar_only` and `l2_even_parity` together define the parity constraint on
            the allowed paths.

            When `polar_only` is `False`, (l2_even_parity` is ignored), no parity
            constraint is applied and the paths are determined solely by the `mode`.

            When `polar_only=True` and `l2_even_parity=False`, we restrict the paths to
            only those that produce polar tensors. In addition, `l2_even_parity=False`
            indicates that l2 is associated with a tensor whose parity is (-1)^l2.
            In this case, given that l1 and l2 are the ranks of two polar tensors,
            their parities are p1 = (-1)^l1 and p2 = (-1)^l2. Then the parity of l3 is
            p3 = p1 * p2 = (-1)^(l1+l2). If we want to stay in the polar tensor space,
            we need p3 = (-1)^l3, which leads to the condition that l1 + l2 - l3 should
            be even.
            Note, l1+l2-l3 is even is always true for the `lite` and `camp` modes by
            construction, so no additional paths are removed in these two modes.

            When `polar_only=True` and `l2_even_parity=False`, we restrict the paths to
            only those that produce polar tensors, In addition, `l2_even_parity=True`
            indicates that is associated with a tensor whose parity is always even,
            i.e. p2=1. In this case, l1 is still associated with a polar tensor,
            whose parity is p1 = (-1)^l1. Then, the output parity is p3 = p1 * p2 =
            (-1)^l1. And, of course, we want p3 = (-1)^l3. This leads to the condition
            that l1 - l3 should be even.

            To sum:
            - polar_only=False: no parity constraint.
            - polar_only=True, l2_even_parity=False: l1 + l2 - l3 should be even. This
              is automatically satisfied for `lite` and `camp` modes.
            - polar_only=True, l2_even_parity=True: l1 - l3 should be even.

        downward: Only paths that go from higher-rank tensors to lower-rank tensors
            are allowed. This can be useful when the goal is to propagate information
            from higher-rank tensors to lower-rank tensors.
            Specifically, the allowed paths satisfy:
            l3 <= max(l1, l2).

            This, together with `mode`, defines how the paths are selected.

    Returns:
        Dictionary of paths from L1 and L2 to L3: {l3: [(l1, l2, l3)]}, where each
        tuple is a valid path from l1 and l2 to l3.
    """

    def is_polar(l1, l2, l, l2_even_parity):
        if l2_even_parity:
            return (l1 + l) % 2 == 0
        else:
            return (l1 + l2 + l) % 2 == 0

    paths = defaultdict(list)

    for l1 in range(L1 + 1):
        for l2 in range(L2 + 1):
            if mode == "full":
                candidate = range(abs(l1 - l2), l1 + l2 + 1)
            elif mode == "level":
                if level is None:
                    raise ValueError("level must be specified when mode is 'level'.")
                candidate = [
                    l
                    for l in range(abs(l1 - l2), l1 + l2 + 1)
                    if l == 0  # Don't apply level restriction for l=0
                    or l1 + l2 <= level
                ]
            elif mode == "level2":
                if level is None:
                    raise ValueError("level must be specified when mode is 'level2'.")
                candidate = [
                    l
                    for l in range(abs(l1 - l2), l1 + l2 + 1)
                    if l == 0  # Don't apply level restriction for l=0
                    or l1 + l2 + l <= level
                ]
            elif mode == "camp":
                candidate = [l1 + l2 - 2 * l1] if l1 <= l2 else []
            elif mode == "lite":
                candidate = [l1 + l2 - 2 * l1] if l1 <= l2 else [l1 + l2 - 2 * l2]
            else:
                raise ValueError(f"Invalid mode: {mode}.")

            for l in candidate:
                # Only keep l in L3
                if l not in L3:
                    continue

                # Only keep downward paths
                if downward and not l <= max(l1, l2):
                    continue

                # If polar_only, only keep paths that produce polar tensors
                if polar_only and not is_polar(l1, l2, l, l2_even_parity):
                    continue

                paths[l].append((l1, l2, l))

    return paths


class BufferList(nn.Module):
    """
    A list of tensors registered as buffers.

    Similar to nn.ParameterList, but for buffers.
    """

    def __init__(self, tensors: list[Tensor]):
        super().__init__()
        for i, tensor in enumerate(tensors):
            self.register_buffer(str(i), tensor)

    def __getitem__(self, idx: int) -> Tensor:
        return getattr(self, str(idx))

    def __len__(self) -> int:
        return len(self._buffers)


if __name__ == "__main__":
    L1 = 2
    L2 = 2
    L3 = list(range(L2 + 1))

    for mode in ["full", "lite", "camp", "level", "level2"]:
        print("\n" + "#" * 80)

        for polar in [False, True]:
            print("\n" + "#" * 40)

            for l2_even_parity in [False, True]:
                if mode in ["level"]:
                    level = L2
                elif mode in ["level2"]:
                    level = L1 + L2 + 1
                else:
                    level = None
                paths = get_paths(
                    L1,
                    L2,
                    L3,
                    mode,
                    level=level,
                    polar_only=polar,
                    l2_even_parity=l2_even_parity,
                    downward=True,
                )

                print("\n" + "#" * 20)
                print(f"mode: {mode}")
                print(f"polar_only: {polar}")
                print(f"l2_even_parity: {l2_even_parity}")
                print(f": {L1}")
                print("Number of paths:", {k: len(v) for k, v in paths.items()})
                pprint(paths)
