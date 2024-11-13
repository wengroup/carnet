"""
Calculations with symbolic tensors in 3D space, including operations with the delta and
epsilon tensors.
"""
import itertools
from collections import Counter, defaultdict
from fractions import Fraction
from typing import Union


class CartesianTensor:
    """
    A general Cartesian tensor T_ij...k.

    This allows for pairs of repeated indices, e.g. ii, jj, etc.

    Args:
        indices: The indices of the tensor.
        factor: The scalar factor multiplied to the tensor, default is 1.
        symbol: The symbol of the tensor, default is "T".
    """

    def __init__(self, indices: str, factor: int | Fraction = 1, symbol: str = "T"):
        self._check_indices(indices)
        self._indices = indices

        if isinstance(factor, int):
            self._factor = Fraction(factor)
        elif isinstance(factor, Fraction):
            self._factor = factor
        else:
            raise ValueError("The `factor` must be a Fraction object")

        self._symbol = symbol

    @property
    def indices(self):
        return self._indices

    @property
    def symbol(self):
        return self._symbol

    @property
    def factor(self):
        return self._factor

    @property
    def rank(self):
        return len(self.indices)

    def permute_indices(
        self, permute: list[int], factor: int | Fraction = 1
    ) -> "CartesianTensor":
        """
        Permute the indices of the tensor.

        For example, if the tensor is T_ijk and permute is [1, 2, 0], the new tensor
        is T_jki.

        Args:
            permute: The new order of the indices.
            factor: The factor to be multiplied to the tensor, default is 1.

        Returns:
            The tensor with permuted indices.
        """
        indices = "".join([self.indices[i] for i in permute])
        if Counter(self.indices) != Counter(indices):
            raise ValueError("The new indices must contain the same indices")

        return self.__class__(indices, factor * self.factor, self.symbol)

    def evaluate(self, mapping: dict[str, str]) -> "Tensors":
        """Evaluate tensor indices to 1, 2, 3."""
        indices = evaluate_indices(self.indices, mapping)
        tensors = [self.__class__(i, self.factor, self.symbol) for i in indices]

        return Tensors(*tensors)

    @staticmethod
    def _check_indices(indices: str):
        """Check indices are repeated at most twice.

        Only check for alphabetic indices, not integer indices. When the indices are evaluated, there can be many repeated values of the same integer.
        """
        for i in indices:
            if i.isalpha() and indices.count(i) > 2:
                raise ValueError("Indices can be repeated at most twice")

    def __contains__(self, item):
        return item in self.indices

    def __iter__(self):
        return iter(self.indices)

    def __getitem__(self, index):
        return self.indices[index]

    def __mul__(self, other: int | Fraction):
        """Multiply the tensor by a scalar."""
        return self.__class__(self.indices, self.factor * other, self.symbol)

    def __rmul__(self, other: int | Fraction):
        return self.__mul__(other)

    def __eq__(self, other):
        idx2pos_self = defaultdict(list)
        idx2pos_other = defaultdict(list)
        for p, i in enumerate(self.indices):
            idx2pos_self[i].append(p)
        for p, i in enumerate(other.indices):
            idx2pos_other[i].append(p)

        # check if the indices are the same, ignoring repeated indices
        idx2pos_self = {k: v for k, v in idx2pos_self.items() if len(v) == 1}
        idx2pos_other = {k: v for k, v in idx2pos_other.items() if len(v) == 1}
        if idx2pos_self != idx2pos_other:
            return False

        return self.symbol == other.symbol and self.factor == other.factor

    def __str__(self):
        if self.factor == Fraction(1):
            return f"{self._symbol}_{self.indices}"
        else:
            return f"({self.factor}) {self._symbol}_{self.indices}"


class Scalar(CartesianTensor):
    """
    Zero rank tensor, a scalar.
    """

    def __init__(self, factor: int | Fraction = 1):
        super().__init__("", factor, "Const")

    def __str__(self):
        return f"{self.symbol}({self.factor})"


class Zero(Scalar):
    """The scalar zero."""

    def __init__(self):
        super().__init__(0)


class Delta(CartesianTensor):
    """
    The Kronecker delta tensor \delta.

    Args:
        indices: The indices of the delta tensor.
        factor: The scalar factor multiplied to the delta tensor, default is 1.
        symbol: The symbol of the delta tensor, default is "δ".
    """

    def __init__(self, indices: str, factor: int | Fraction = 1, symbol: str = "δ"):
        assert len(indices) == 2, "The delta tensor must have two indices"
        super().__init__(indices, factor, symbol)

    def evaluate(self, mapping: dict[str, str]):
        # check mapping consists of all indices for epsilon tensor
        assert set(self.indices).issubset(set(mapping.keys())), "Invalid mapping keys"

        assert set(mapping.values()).issubset({"1", "2", "3"}), "Invalid mapping values"

        if mapping[self[0]] == mapping[self[1]]:
            return Scalar(3)
        else:
            return Zero()

    def __eq__(self, other):
        # should not compare symbol, but just make sure it is a Delta tensor
        if not isinstance(other, Delta):
            return False

        if not self.factor == other.factor:
            return False

        if self.indices != other.indices and self.indices != other.indices[::-1]:
            return False

        return True


class Epsilon(CartesianTensor):
    """
    The Levi-Civita tensor \epsilon.

    Args:
        indices: The indices of the epsilon tensor.
        factor: The scalar factor multiplied to the epsilon tensor, default is 1.
        symbol: The symbol of the epsilon tensor, default is "ε".
    """

    def __init__(self, indices: str, factor: int | Fraction = 1, symbol: str = "ε"):
        assert len(indices) == 3, "The epsilon tensor must have three indices"
        super().__init__(indices, factor, symbol)

    def evaluate(self, mapping: dict[str, str]) -> Union["CartesianTensor", "Tensors"]:
        """
        Evaluate tensor indices to 1, 2, 3.

        e_123, e_231, e_312 = 1
        e_132, e_213, e_321 = -1
        others = 0
        """

        # check mapping consists of all indices for epsilon tensor
        assert set(self.indices).issubset(set(mapping.keys())), "Invalid mapping keys"

        assert set(mapping.values()).issubset({"1", "2", "3"}), "Invalid mapping values"

        indices = "".join(mapping[i] for i in self)

        if indices in ["123", "231", "312"]:
            return Scalar(self.factor)
        elif indices in ["132", "213", "321"]:
            return Scalar(-1 * self.factor)
        else:
            return Zero()

    def __eq__(self, other):
        # should not compare symbol, but just make sure it is a Epsilon tensor
        if not isinstance(other, Epsilon):
            return False

        if not self.factor == other.factor:
            return False

        # even permutations of indices are equal
        indices = other.indices
        if (
            self.indices != indices
            and self.indices != indices[1] + indices[2] + indices[0]
            and self.indices != indices[2] + indices[0] + indices[1]
        ):
            return False

        return True


class TensorProduct:
    """
    A representation of a tensor product of multiple tensors.

    Args:
        tensors: The constituting tensors.
        factor: Additional factor multiplied to the tensor product. Each tensor in the
            product can have its own factor. So, the overall factor is the product of
            the factors of the constituting tensors and this factor.
        combine_scalars: If True, combine scalars in the tensor product. For example, all
            scalars will be combined into the factor of the tensor product, and the scalars
            will be removed from the tensor product. Default is True.
    """

    def __init__(
        self,
        *tensors: CartesianTensor | Epsilon | Delta | Scalar,
        factor: int | Fraction = 1,
        combine_scalars: bool = True,
    ):
        self.combine_scalars = combine_scalars

        if not combine_scalars:
            self._factor = factor
            self._tensors = list(tensors)
        else:
            # get overall factor
            for t in tensors:
                factor *= t.factor
            self._factor = factor

            if self._factor == 0:
                self._tensors = [Zero()]

            else:
                # set the factor of the constituting tensors to 1
                self._tensors = []
                for t in tensors:
                    if isinstance(t, Scalar):
                        pass  # scalars already been included in the factor
                    else:
                        self._tensors.append(t.__class__(t.indices, 1, t.symbol))

    @property
    def factor(self):
        """The overall factor of the tensor product."""
        return self._factor

    @property
    def components(self):
        """Constituting tensors of the product, without considering the factor."""
        return self._tensors

    @property
    def indices(self):
        """The indices of the tensor product."""
        return "".join([t.indices for t in self._tensors])

    def permute_indices(
        self, permute: list[int], factor: int | Fraction = 1
    ) -> "TensorProduct":
        """
        Permute the indices of the tensor product.

        For example,
        if the tensor is D_ab T_ijk and permute is [2,4,0,1,3], the new tensor is
        D_ik T_abj.

        Args:
            permute: The new order of the indices.
            factor: Additional factor to be multiplied to the tensor, default is 1.

        Returns:
            The tensor product with permuted indices.
        """

        indices = self.indices

        i = 0
        tensors = []
        for t in self._tensors:
            perm = permute[i : i + len(t.indices)]
            permuted_indices = "".join([indices[p] for p in perm])
            nt = t.__class__(permuted_indices, t.factor, t.symbol)
            tensors.append(nt)
            i += len(t.indices)

        return TensorProduct(*tensors, factor=factor * self.factor)

    def evaluate(self, mapping: dict[str, str]) -> "Tensors":
        """
        Evaluate tensor indices to 1, 2, 3.
        """
        # evaluate indices of all constituting tensors
        all_indices = evaluate_indices(self.indices, mapping, strict=True)

        tensor_product = []
        for indices in all_indices:
            # new mapping based on the evaluated indices
            new_mapping = dict(zip(self.indices, indices))

            # evaluate each constituting tensor
            evaluated = []
            for t in self._tensors:
                out = t.evaluate(new_mapping)
                if isinstance(out, Tensors):
                    evaluated.append(list(out._tensors))
                else:
                    evaluated.append([out])

            for tensors in itertools.product(*evaluated):
                tp = TensorProduct(*tensors, factor=self.factor)
                tensor_product.append(tp)

        return Tensors(*tensor_product)

    def __eq__(self, other: Union[CartesianTensor, "TensorProduct"]):
        # TODO, we just implement the case that the constituting tensors are the same
        #  and in the same order. Of course, this is not general.

        if len(self) != len(other):
            return False

        if self.factor != other.factor:
            return False

        # compare symbol and indices of the constituting tensors
        for x, y in zip(self._tensors, other._tensors):
            if x != y:
                return False

        return True

    def __mul__(self, other: int | Fraction):
        """Multiply the tensor by a scalar."""
        return self.__class__(*self._tensors, factor=self.factor * other)

    def __rmul__(self, other: int | Fraction):
        return self.__mul__(other)

    def __iter__(self):
        return iter(self._tensors)

    def __getitem__(self, item):
        return self._tensors[item]

    def __len__(self):
        return len(self._tensors)

    def __str__(self):
        rep = ""
        for t in self._tensors:
            # scalars will be included in the factor, so we skip them here
            if not isinstance(t, Scalar):
                rep += f" {t.symbol}_{t.indices}"

        return f"({self.factor}){rep}"


class Tensors:
    """A linear combination of multiple Cartesian tensors."""

    def __init__(self, *tensors: CartesianTensor | Delta | Epsilon | TensorProduct):
        self._tensors = tensors

    def evaluate(self, mapping: dict[str, str]) -> "Tensors":
        """Evaluate tensor indices to 1, 2, 3."""

        evaluated = []
        for t in self._tensors:
            e = t.evaluate(mapping)
            if isinstance(e, Tensors):
                evaluated.extend(e)
            else:
                evaluated.append(e)

        return self.__class__(*evaluated)

    @property
    def components(self):
        """The constituting tensor products."""
        return self._tensors

    def to_str_list(self, including_zero: bool = False) -> list[str]:
        """
        Convert the tensors to string representation.

        Args:
            including_zero: If True, include zero tensors in the output.
        """
        return [str(t) for t in self._tensors if including_zero or t.factor != 0]

    def __eq__(self, other: "Tensors"):
        # TODO, we just implement the case that the constituting tensors are the same
        #  and in the same order. Of course, this is not general.

        if len(self) != len(other):
            return False

        for x, y in zip(self._tensors, other._tensors):
            if x != y:
                return False

        return True

    def __len__(self):
        return len(self._tensors)

    def __iter__(self):
        return iter(self._tensors)

    def __getitem__(self, item):
        return self._tensors[item]

    def __add__(self, other: "Tensors"):
        return Tensors(*self._tensors, *other._tensors)

    def __mul__(self, other: int | Fraction):
        """Multiply the tensor by a scalar."""
        return Tensors(*[t * other for t in self._tensors])

    def __rmul__(self, other: int | Fraction):
        return self.__mul__(other)

    def __str__(self):
        str_rep = self.to_str_list(including_zero=False)

        return "Tensors(\n   " + "\n + ".join(str_rep) + "\n)"


def multiply(
    *tensors: CartesianTensor | TensorProduct, factor: int | Fraction = 1
) -> TensorProduct:
    """
    Multiple a list of tensors or tensor products to create a new tensor product.

    Args:
        *tensors: the tensors or tensor products to multiply.
        factor: Additional factor to be multiplied to the tensor product, default is 1.

    Returns:
        The new tensor product.
    """
    new_tensors = []
    factor = Fraction(factor)
    for t in tensors:
        if isinstance(t, CartesianTensor):
            new_tensors.append(t)
        elif isinstance(t, TensorProduct):
            new_tensors.extend(t.components)
            factor *= t.factor
        else:
            raise ValueError("Unexpected type")

    tp = TensorProduct(*new_tensors, factor=factor)

    return tp


def contract_with_delta(delta: Delta, tensor: CartesianTensor) -> CartesianTensor:
    """
    Contract a tensor with a delta tensor.

    For example,
    \delta_ij T_ijk -> T_iik
    \delta_ai T_ijk -> T_ajk

    Args:
        delta: The delta tensor.
        tensor: A Cartesian tensor.

    Returns:
        The contracted tensor.
    """
    # check at least one of the indices is in common
    if not (set(tensor.indices) & set(delta.indices)):
        raise ValueError("Delta tensor does not have common indices with the tensor")

    for p, i in enumerate(delta):
        if i in tensor:
            other = delta[1] if p == 0 else delta[0]
            return tensor.__class__(
                tensor.indices.replace(i, other), tensor.factor, tensor.symbol
            )

    raise ValueError("Delta tensor does not have common indices with the tensor")


def contract_with_epsilon(epsilon: Epsilon, tensor: CartesianTensor) -> TensorProduct:
    """
    Contract a tensor with an epsilon tensor.

    \epsilon_aij T_ijk...n
    \epsilon_abi T_ijk...n

    Args:
        epsilon: The epsilon tensor.
        tensor: A Cartesian tensor.

    Returns:
        The contracted tensor.
    """
    # check at least one of the indices is in common
    if not (set(tensor.indices) & set(epsilon.indices)):
        raise ValueError("Epsilon tensor does not have common indices with the tensor")
    return TensorProduct(epsilon, tensor)


def contract_epsilon_delta(epsilon: Epsilon, delta: Delta):
    """
    Contract an epsilon tensor with a delta tensor.

    For example,
    \epsilon_ijk \delta_ij = \epsilon_iik = 0
    \epsilon_ijk \delta_il = \epsilon_ljk

    At least one of the indices must be repeated in the delta tensor

    Args:
        epsilon: The epsilon tensor, given by three indices.
        delta: The delta tensor, given by a pair of indices.

    Returns:
        The contracted tensor.
    """
    if len(set(epsilon) & set(delta)) == 2:
        return Zero()

    return contract_with_delta(delta, epsilon)


def contract_two_epsilon(epsilon1: Epsilon, epsilon2: Epsilon):
    """
    Contract two epsilon tensors.

    This implements:
    1. e_ijk e_pqk = d_ip d_jq - d_iq d_jp
    2. e_ijk e_pjk = 2 d_ip
    3. e_ijk e_ijk = 6

    Args:
        epsilon1: The first epsilon tensor.
        epsilon2: The second epsilon tensor.

    Returns:
        The contracted delta tensor.
    """

    def canonicalize_one(eps, idx):
        """
        Canonicalize the order of the indices.

        Does not change relative order of the three indices, but put the provided index
        at the last position.

        For example,
            (ijk, k) -> ijk
            (ijk, j) -> kij
            (ijk, i) -> jki
        """
        if idx == eps[0]:
            indices = eps[1] + eps[2] + eps[0]
            return Epsilon(indices, eps.factor)
        elif idx == eps[1]:
            indices = eps[2] + eps[0] + eps[1]
            return Epsilon(indices, eps.factor)
        else:
            return eps

    def canonicalize_two(eps, idx1, idx2):
        """
        Canonicalize the order of the indices.

        Put idx1 at the second position, idx2 at the third position.
        It relative order of the indices is changed, the sign is flipped.

        For example,
        (ijk, i, j) -> kij
        (ijk, j, i) -> -kij
        (ijk, j, k) -> ijk
        (ijk, k, j) -> -ijk
        (ijk, k, i) -> jki
        (ijk, i, k) -> -jki
        """
        if idx1 == eps[0] and idx2 == eps[1]:
            indices = eps[2] + eps[0] + eps[1]
            sign = 1

        elif idx1 == eps[1] and idx2 == eps[2]:
            indices = eps[0] + eps[1] + eps[2]
            sign = 1

        elif idx1 == eps[2] and idx2 == eps[0]:
            indices = eps[1] + eps[2] + eps[0]
            sign = 1

        elif idx1 == eps[1] and idx2 == eps[0]:
            indices = eps[2] + eps[1] + eps[0]
            sign = -1

        elif idx1 == eps[2] and idx2 == eps[1]:
            indices = eps[0] + eps[2] + eps[1]
            sign = -1

        elif idx1 == eps[0] and idx2 == eps[2]:
            indices = eps[1] + eps[0] + eps[2]
            sign = -1

        else:
            raise ValueError("Invalid indices")

        return Epsilon(indices, sign * eps.factor)

    # get number of repeated indices
    repeated = set(epsilon1) & set(epsilon2)

    if len(repeated) == 3:
        return Scalar(6)
    elif len(repeated) == 2:
        idx1, idx2 = sorted(repeated)
        eps1 = canonicalize_two(epsilon1, idx1, idx2)
        eps2 = canonicalize_two(epsilon2, idx1, idx2)
        return Delta(eps1[0] + eps2[0], 2 * eps1.factor * eps2.factor)

    elif len(repeated) == 1:
        idx = repeated.pop()
        eps1 = canonicalize_one(epsilon1, idx)
        eps2 = canonicalize_one(epsilon2, idx)
        d1 = Delta(eps1[0] + eps2[0])
        d2 = Delta(eps1[1] + eps2[1])
        d3 = Delta(eps1[0] + eps2[1])
        d4 = Delta(eps1[1] + eps2[0])
        return Tensors(TensorProduct(d1, d2), TensorProduct(d3, d4, factor=-1))

    else:
        raise ValueError("No repeated indices")


def symmetrize(tensor: CartesianTensor | TensorProduct, indices: str = None) -> Tensors:
    """
    Symmetrize a tensor.

    Args:
        tensor: A Cartesian tensor.
        indices: The indices to symmetrize over. If None, all non-repeated indices are symmetrized.

    Returns:
        A list of tensors, each with a different permutation of the indices, each tensor
        is normalized by the number of permutations.
    """

    if indices is None:
        indices = [i for i, c in Counter(tensor.indices).items() if c == 1]
    else:
        # check provided indices are not repeated in the tensor
        for i in indices:
            if tensor.indices.count(i) != 1:
                raise ValueError(f"Index {i} must appear exactly once in the tensor")

    moveable_pos = [i for i, x in enumerate(tensor.indices) if x in indices]

    all_tensors = []
    permutations = list(itertools.permutations(moveable_pos))
    for perm in permutations:
        # candidate permute
        permute = list(range(len(tensor.indices)))
        # update permute positions
        for i, p in zip(moveable_pos, perm):
            permute[i] = p

        t = tensor.permute_indices(permute, factor=Fraction(1, len(permutations)))
        all_tensors.append(t)

    return Tensors(*all_tensors)


def evaluate_indices(
    indices: str, mapping: dict[str, str | int], strict: bool = False
) -> list[str]:
    """
    Evaluate the given indices to 1, 2, and 3.

    1. Repeat indices are automatically expanded to 11, 22, 33.
    2. For non-repeated indices, it will be expanded according to the mapping.
       If a mapping for an index is not provided, it will not be expanded.

    Args:
        indices: the indices to evaluate.
        mapping: {i:v} index value pairs to use in the evaluation.
            Examples: {"i": "1", "j": "2", "k": "3"}
        strict: If True, raise an error mapping is provided for an index not in the
            indices.

    Returns:
        A list of evaluated indices.
    """
    # convert mapping values to strings
    mapping = {k: str(v) for k, v in mapping.items()}

    # checking all mapping keys are in the indices
    if strict and not set(mapping.keys()).issubset(set(indices)):
        raise ValueError("All mapping keys should be in the indices")

    # check mapping values are 1, 2, or 3
    if not set(mapping.values()).issubset({"1", "2", "3"}):
        raise ValueError("All mapping values should be 1, 2, or 3")

    # find all repeated indices
    count = Counter(indices)

    single_indices = []
    double_indices = []
    for i, c in count.items():
        if c == 1:
            single_indices.append(i)
        elif c == 2:
            double_indices.append(i)
        else:
            raise ValueError("Indices can be repeated at most twice")

    # expand single indices according to the mapping
    indices = list(indices)
    for x in single_indices:
        if x in mapping:
            indices[indices.index(x)] = mapping[x]
    indices = "".join(indices)

    # expand double indices to 1, 2, 3
    indices = [indices]
    new_indices = []
    for i in double_indices:
        for t in indices:
            for j in range(1, 4):
                new_indices.append(t.replace(i, str(j)))
        indices = new_indices
        new_indices = []

    return indices


def is_zero(tensors: Tensors) -> bool:
    """
    Check whether a linear combination of tensors is zero.
    """
    # TODO, for now, we just check if the str representation of the positive ones
    #  and the negative ones are the same
    positive = []
    negative = []
    for t in tensors:
        if t.factor == 0:
            continue
        elif t.factor > 0:
            positive.append(t)
        else:
            negative.append(t)

    # flip the sign of the negative ones
    negative = [-1 * t for t in negative]

    pos_count = Counter([str(t) for t in positive])
    neg_count = Counter([str(t) for t in negative])

    return pos_count == neg_count


def simplify(tp: TensorProduct) -> Tensors:
    """
    Simplify a tensor product by apply delta and epsilon rules.

    For example,
    d_ij e_imn d_nq T_qpr -> e_jmq T_qpr
    """

    def _simplify(product: TensorProduct) -> tuple[TensorProduct | Tensors, bool]:
        """
        Simplify a tensor product by apply delta and epsilon rules.

        Returns:
            out: The simplified tensor product or tensors.
            performed: If any simplification is performed.
        """
        # Simplify product two epsilon tensors
        epsilon_pos = [i for i, t in enumerate(product) if isinstance(t, Epsilon)]
        if len(epsilon_pos) >= 2:
            i = epsilon_pos[0]
            for j in epsilon_pos[1:]:
                # check if they share at least one index
                if set(product[i].indices) & set(product[j].indices):
                    out = contract_two_epsilon(product[i], product[j])

                    # Create simplified tensors, without the two epsilon tensors
                    # and with the contracted tensor placed at the beginning
                    new_tensors = [t for p, t in enumerate(product) if p not in [i, j]]

                    # three identical indices, resulting in a scalar
                    if isinstance(out, Scalar):
                        return (
                            TensorProduct(out, *new_tensors, factor=product.factor),
                            True,
                        )
                    # two identical indices, resulting in a delta tensor
                    elif isinstance(out, Delta):
                        return (
                            TensorProduct(out, *new_tensors, factor=product.factor),
                            True,
                        )
                    # one identical index, resulting in linear combination of tensor
                    # products of delta tensors e_ijk e_ilm = d_jl d_km - d_jm d_kl
                    elif isinstance(out, Tensors):
                        linear_comb = []
                        for tp in out:
                            new_tp = TensorProduct(
                                *tp.components,
                                *new_tensors,
                                factor=tp.factor * product.factor,
                            )
                            linear_comb.append(new_tp)

                        return Tensors(*linear_comb), True

                    else:
                        raise ValueError("Invalid output")

        # simplify product with delta

        # check if there are delta tensors
        delta_pos = [i for i, t in enumerate(product) if isinstance(t, Delta)]

        for i in delta_pos:
            delta: Delta = product[i]
            for j, t in enumerate(product):
                if j == i:
                    continue

                # check if they share at least one index
                if set(delta.indices) & set(t.indices):
                    if isinstance(t, Epsilon):
                        out = contract_epsilon_delta(t, delta)
                    else:
                        out = contract_with_delta(delta, t)

                    # Create simplified tensors, without the two epsilon tensors
                    # and with the contracted tensor placed at the beginning
                    new_tensors = [t for p, t in enumerate(product) if p not in [i, j]]

                    return TensorProduct(out, *new_tensors, factor=product.factor), True

        return product, False

    # Iteratively simplify the tensor product
    performed = True
    simplified = Tensors(tp)
    while performed:
        # positions in new_simplified that are results of a double epsilon contraction
        # This leads to a sum of two tensor products of delta tensors, and we need
        # to expand it to be produced with others
        double_epsilon = None
        double_epsilon_pos = None
        new_simplified = []
        performed = []
        for i, tp in enumerate(simplified):
            sim, perf = _simplify(tp)
            new_simplified.append(sim)
            performed.append(perf)
            if isinstance(sim, Tensors):
                double_epsilon_pos = i
                double_epsilon = sim

        # expand double epsilon contraction (two terms) with others
        if double_epsilon is not None:
            linear_comb = []
            for de in double_epsilon:
                # list of tensor products
                comb = new_simplified.copy()
                comb[double_epsilon_pos] = de
                new_tp = multiply(*comb)
                linear_comb.append(new_tp)
        else:
            linear_comb = new_simplified

        # prepare for the next iteration
        performed = any(performed)
        simplified = Tensors(*linear_comb)

    return simplified


if __name__ == "__main__":
    ###
    # Example 1
    # check e_aij T_ijkl, e_aij T_ikjl, and e_aij T_kijl are linearly dependent

    # basic tensors
    e = Epsilon("aij")
    tp1 = contract_with_epsilon(e, CartesianTensor("ijkl"))
    tp2 = contract_with_epsilon(e, CartesianTensor("ikjl"))
    tp3 = contract_with_epsilon(e, CartesianTensor("kijl"))

    # symmetrize the tensors
    s1 = symmetrize(tp1, indices="akl")
    s2 = symmetrize(tp2, indices="akl")
    s3 = symmetrize(tp3, indices="akl")

    tensors = s1 + -1 * s2 + s3

    evaluated = tensors.evaluate(
        {
            "a": "1",
            "i": "2",
            "j": "3",
            "k": "2",
            "l": "3",
        }
    )

    evaluated_non_zero = Tensors(*[t for t in evaluated if t.factor != 0])

    out = is_zero(evaluated_non_zero)

    print("Tensors", tensors)
    print("number of non-zeros:", len(evaluated_non_zero))
    print("evaluated non-zeros", evaluated_non_zero)
    print("Dependence:", out)
