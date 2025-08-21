from torch import Tensor, nn


class MLP(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_features: list[int] = None,
        activation: nn.Module = nn.SiLU(),
        out_activation: bool = False,
    ):
        """
        MLP with SiLU activation.

        Total number of layers is len(hidden_features) + 1.

        Args:
            in_features:
            out_features:
            hidden_features: List of hidden layer sizes. If None, no hidden layers.
            activation: Activation function. Default is SiLU.
            out_activation: Whether to apply activation to the output layer.
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.hidden_features = hidden_features
        self.out_activation = out_activation

        sizes = [in_features]
        if hidden_features is not None:
            sizes += list(hidden_features)
        sizes += [out_features]

        layers = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1]))
            layers.append(activation)

        if not out_activation:
            layers.pop()

        self.layers = nn.Sequential(*layers)

    def forward(self, input: Tensor) -> Tensor:
        return self.layers(input)
