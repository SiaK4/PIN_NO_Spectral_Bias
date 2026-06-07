"""
DeepONet.

Causality is enforced through zero-padded data preprocessing.
"""

from typing import List, Optional

import torch
import torch.nn as nn

from ..evaluation.constants import (
    BRANCH_ACTIVATION_DEFAULT,
    DROPOUT_DEFAULT,
    SIREN_W0_DEFAULT,
    TRUNK_ACTIVATION_DEFAULT,
)
from .base_deepo import BaseDeepO

try:
    from siren_pytorch import SirenNet

    SIREN_AVAILABLE = True
except ImportError:
    SIREN_AVAILABLE = False
    SirenNet = None


class ReQU(nn.Module):
    """
    ReQU activation: ReLU squared (ReLU(x)²).

    Smoother than ReLU for gradient flow.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(x) ** 2


class SinActivation(nn.Module):
    """Plain sin(x) activation (no SIREN w0 initialization), for ablation comparison."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(x)


# Activation function registry for MLP
_ACTIVATION_REGISTRY = {
    "requ": ReQU,
    "tanh": nn.Tanh,
    "relu": nn.ReLU,
    "sin": SinActivation,
    "gelu": nn.GELU,
}


class MLP(nn.Module):
    """
    Multi-layer perceptron with configurable hidden layers and activation functions.

    Supports:
    - 'requ': ReQU (ReLU²) activation
    - 'siren': Sinusoidal activation (requires siren-pytorch)
    - 'tanh': Tanh activation (stable for operator learning)
    - 'relu': ReLU activation
    - 'gelu': GELU activation (smooth approximation of ReLU)
    - 'sin': Plain sin(x) activation
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_layers: List[int],
        activation: str,
        dropout: float = DROPOUT_DEFAULT,
        siren_w0: float = SIREN_W0_DEFAULT,
    ):
        """
        Initialize MLP.

        Args:
            in_features: Input dimension
            out_features: Output dimension
            hidden_layers: List of hidden layer sizes
            activation: Activation function type (required). Options:
                       'requ', 'tanh', 'relu', 'siren', 'sin', 'gelu'
            dropout: Dropout rate
            siren_w0: SIREN w0 parameter (initial frequency scale)

        Raises:
            ValueError: If activation type is not supported
            ImportError: If siren activation is requested but siren-pytorch not installed
        """
        super().__init__()

        self.activation_type = activation.lower()
        self.dropout_rate = dropout
        self.siren_w0 = siren_w0

        # For SIREN, use the SirenNet module from siren-pytorch
        if self.activation_type == "siren":
            if not SIREN_AVAILABLE:
                raise ImportError(
                    "siren-pytorch is required for SIREN activation. "
                    "Install with: pip install siren-pytorch"
                )
            self.network = SirenNet(
                dim_in=in_features,
                dim_hidden=hidden_layers[0] if hidden_layers else 256,
                dim_out=out_features,
                num_layers=len(hidden_layers) + 1,
                final_activation=nn.Identity(),
                w0_initial=siren_w0,
                dropout=dropout,
            )
        else:
            layers = []
            prev_size = in_features

            activation_fn = self._get_activation()

            for hidden_size in hidden_layers:
                layers.append(nn.Linear(prev_size, hidden_size))
                layers.append(activation_fn())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
                prev_size = hidden_size

            layers.append(nn.Linear(prev_size, out_features))

            self.network = nn.Sequential(*layers)

    def _get_activation(self) -> type[nn.Module]:
        """
        Get activation module based on activation type.

        Returns:
            Activation module class

        Raises:
            ValueError: If activation type is not supported
        """
        if self.activation_type not in _ACTIVATION_REGISTRY:
            raise ValueError(
                f"Unsupported activation: '{self.activation_type}'. "
                f"Supported: {list(_ACTIVATION_REGISTRY.keys()) + ['siren']}"
            )
        return _ACTIVATION_REGISTRY[self.activation_type]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through MLP."""
        return self.network(x)


class DeepONet(BaseDeepO):
    """
    DeepONet.

    Architecture:
        - Branch network: Encodes windowed input function (4000 timesteps) → latent vector
        - Trunk network: Encodes time coordinate (scalar) → latent vector
        - Combination: Element-wise product + sum → scalar output

    Causality:
        Enforced through per-timestep windowing with zero-padding.
        At timestep t, branch receives only inputs from times [0, ..., t].

    Forward signatures (inherited from BaseDeepO):
        - forward_per_timestep(input, time_coord) → [batch, 1]
        - forward(input) → [batch, 1, sensor_dim]
        - forward_causal_sequence(input) → [batch, 1, sensor_dim]

    """

    def __init__(
        self,
        sensor_dim: int = 4000,
        latent_dim: int = 100,
        branch_layers: Optional[List[int]] = None,
        trunk_layers: Optional[List[int]] = None,
        branch_activation: str = BRANCH_ACTIVATION_DEFAULT,
        trunk_activation: str = TRUNK_ACTIVATION_DEFAULT,
        dropout: float = DROPOUT_DEFAULT,
        siren_w0: float = SIREN_W0_DEFAULT,
        init_type: str = "pytorch_default",
    ):
        """Initialize DeepONet branch/trunk networks."""
        super().__init__()
        self.init_type = init_type

        self.sensor_dim = sensor_dim
        self.latent_dim = latent_dim
        self.branch_layers = branch_layers or [120, 120]
        self.trunk_layers = trunk_layers or [120, 120]
        self.branch_activation = branch_activation
        self.trunk_activation = trunk_activation
        self.dropout = dropout
        self.siren_w0 = siren_w0

        # Causality enforced through zero-padded input data
        branch_kwargs = {
            "in_features": sensor_dim,
            "out_features": latent_dim,
            "hidden_layers": self.branch_layers,
            "activation": branch_activation,
            "dropout": dropout,
        }
        if branch_activation.lower() == "siren":
            branch_kwargs["siren_w0"] = siren_w0
        self.branch = MLP(**branch_kwargs)

        trunk_kwargs = {
            "in_features": 1,
            "out_features": latent_dim,
            "hidden_layers": self.trunk_layers,
            "activation": trunk_activation,
            "dropout": dropout,
        }
        if trunk_activation.lower() == "siren":
            trunk_kwargs["siren_w0"] = siren_w0
        self.trunk = MLP(**trunk_kwargs)

        self.output_layer = nn.Linear(latent_dim, 1, bias=False)

        if self.init_type != "pytorch_default":
            self._apply_init(self.init_type)

    def _apply_init(self, init_type: str) -> None:
        """
        Apply specified initialization to non-SIREN layers.

        Args:
            init_type: One of 'xavier', 'kaiming', 'pytorch_default'
                      - xavier: Xavier/Glorot uniform (best for tanh/sigmoid)
                      - kaiming: Kaiming/He uniform (best for relu)
                      - pytorch_default: No-op, use PyTorch's default

        SIREN layers always keep their own w0-based initialization.
        """
        if init_type == "pytorch_default":
            return

        if self.branch_activation != "siren":
            for m in self.branch.modules():
                if isinstance(m, nn.Linear):
                    self._init_linear(m, init_type)

        if self.trunk_activation != "siren":
            for m in self.trunk.modules():
                if isinstance(m, nn.Linear):
                    self._init_linear(m, init_type)

        self._init_linear(self.output_layer, init_type)

    def _init_linear(self, layer: nn.Linear, init_type: str) -> None:
        """
        Initialize a single Linear layer with the specified initialization.

        Args:
            layer: nn.Linear layer to initialize
            init_type: 'xavier' or 'kaiming'
        """
        if init_type == "xavier":
            nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
        elif init_type == "kaiming":
            nn.init.kaiming_uniform_(layer.weight, nonlinearity="relu")
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
        else:
            raise ValueError(
                f"Unknown init_type: {init_type}. "
                "Options: 'xavier', 'kaiming', 'pytorch_default'"
            )

    def _apply_output_layer(
        self, combined: torch.Tensor, is_batched_3d: bool = False
    ) -> torch.Tensor:
        """
        Apply linear output layer .

        Args:
            combined: Combined branch*trunk tensor
                     - 2D: [batch, latent_dim] for per-timestep modes
                     - 3D: [batch, seq_len, latent_dim] for full-sequence (is_batched_3d=True)
            is_batched_3d: Whether input is 3D (full-sequence mode)

        Returns:
            Output tensor:
            - 2D input: [batch, 1]
            - 3D input: [batch, 1, seq_len]
        """
        output = self.output_layer(combined)

        if is_batched_3d:
            return output.squeeze(-1).unsqueeze(1)
        return output  # [batch, 1]
