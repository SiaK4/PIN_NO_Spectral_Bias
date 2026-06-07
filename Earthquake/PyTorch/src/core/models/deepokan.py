"""
DeepOKAN.

DeepOKAN replaces the MLP layers in DeepONet with Chebyshev KAN (Kolmogorov-Arnold
Network using Chebyshev polynomial basis) while maintaining full API compatibility.

Causality is enforced through zero-padded data preprocessing.
"""

from typing import List, Optional

import torch
import torch.nn as nn

from .base_deepo import BaseDeepO


def chebyshev_polynomials_vectorized(x: torch.Tensor, degree: int = 7) -> torch.Tensor:
    """
    Compute Chebyshev polynomials T_0(x) through T_degree(x) using recurrence.

    Args:
        x: Input tensor of shape [...] (any shape, typically [batch, in_features])
        degree: Maximum polynomial degree (default 7 for T0-T7)

    Returns:
        Tensor of shape [..., degree+1] containing [T_0(x), T_1(x), ..., T_degree(x)]
    """
    # Avoid in-place operations to preserve gradient computation.
    polynomials = []

    # T_0(x) = 1
    polynomials.append(torch.ones_like(x))

    if degree >= 1:
        # T_1(x) = x
        polynomials.append(x)

    if degree >= 2:
        # Pre-compute 2x for recurrence
        two_x = 2.0 * x

        # Use recurrence: T_{n+1}(x) = 2x*T_n(x) - T_{n-1}(x)
        for n in range(1, degree):
            T_next = two_x * polynomials[n] - polynomials[n - 1]
            polynomials.append(T_next)

    # Stack along new last dimension: [...] -> [..., degree+1]
    return torch.stack(polynomials, dim=-1)


class _ChebyshevKANLayer(nn.Module):
    """
    Single Chebyshev KAN layer.

    Transforms input through:
    1. tanh activation to bound input to [-1, 1]
    2. Chebyshev polynomial basis expansion (T0-T7)
    3. Learned linear combination via einsum

    Args:
        in_features: Input dimension
        out_features: Output dimension
        degree: Chebyshev polynomial degree (default 7 for T0-T7)

    Weight shape: [in_features, out_features, degree+1]
    Initialization: Kaiming with fan_in = in_features * (degree + 1)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        degree: int = 7,
        init_type: str = "pytorch_default",
    ):
        """
        Initialize ChebyshevKANLayer.

        Args:
            in_features: Number of input features
            out_features: Number of output features
            degree: Chebyshev polynomial degree (default 7 for T0-T7)
            init_type: Initialization type for Chebyshev coefficients. Options:
                      - 'kaiming': Kaiming/He init with std=sqrt(2/fan_in) (default, optimal for ReLU-like)
                      - 'xavier': Xavier/Glorot init with std=sqrt(2/(fan_in+fan_out)) (optimal for tanh)
                      - 'pytorch_default': PyTorch Linear default (Kaiming uniform with a=sqrt(5))
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.degree = degree
        self.init_type = init_type

        fan_in = in_features * (degree + 1)
        fan_out = out_features * (degree + 1)

        if init_type == "kaiming":
            std = (2.0 / fan_in) ** 0.5
            self.weight = nn.Parameter(
                torch.normal(0.0, std, size=(in_features, out_features, degree + 1))
            )
        elif init_type == "xavier":
            std = (2.0 / (fan_in + fan_out)) ** 0.5
            self.weight = nn.Parameter(
                torch.normal(0.0, std, size=(in_features, out_features, degree + 1))
            )
        elif init_type == "pytorch_default":
            bound = 1.0 / (fan_in**0.5)
            self.weight = nn.Parameter(
                torch.empty(in_features, out_features, degree + 1).uniform_(
                    -bound, bound
                )
            )
        else:
            raise ValueError(
                f"Unknown init_type: {init_type}. Must be 'kaiming', 'xavier', or 'pytorch_default'"
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through a Chebyshev KAN layer.

        Args:
            x: Input tensor of shape [batch, in_features]

        Returns:
            Output tensor of shape [batch, out_features]
        """
        x = torch.tanh(x)
        x_cheby = chebyshev_polynomials_vectorized(x, degree=self.degree)

        # 'bid,iod->bo': [batch, in_features, degree+1] @ [in_features, out_features, degree+1]
        # -> [batch, out_features]
        x_out = torch.einsum("bid,iod->bo", x_cheby, self.weight)

        return x_out


class _ChebyshevKANNetwork(nn.Module):
    """
    Multi-layer Chebyshev KAN network.

    Stacks multiple _ChebyshevKANLayer modules with optional dropout.

    Args:
        layer_dims: List of layer dimensions [input_dim, hidden1, hidden2, ..., output_dim]
        degree: Chebyshev polynomial degree (default 7)
        dropout: Dropout probability between hidden layers (default 0.0)

    Example:
        >>> net = _ChebyshevKANNetwork([4000, 75, 75, 75, 75, 75, 75, 75, 100])
        >>> x = torch.randn(16, 4000)
        >>> y = net(x)  # [16, 100]
    """

    def __init__(
        self,
        layer_dims: List[int],
        degree: int = 7,
        dropout: float = 0.0,
        init_type: str = "pytorch_default",
    ):
        """
        Initialize ChebyshevKANNetwork.

        Args:
            layer_dims: List of layer dimensions [input_dim, hidden1, hidden2, ..., output_dim]
            degree: Chebyshev polynomial degree (default 7)
            dropout: Dropout probability between hidden layers (default 0.0)
            init_type: Initialization type for all KAN layers. Options:
                      - 'kaiming': Kaiming/He init
                      - 'xavier': Xavier/Glorot init
                      - 'pytorch_default': PyTorch default
        """
        super().__init__()
        self.layer_dims = layer_dims
        self.degree = degree
        self.init_type = init_type

        self.layers = nn.ModuleList()
        for i in range(len(layer_dims) - 1):
            self.layers.append(
                _ChebyshevKANLayer(
                    layer_dims[i], layer_dims[i + 1], degree, init_type=init_type
                )
            )

        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the KAN stack.

        Expects pre-normalized input.

        Args:
            x: Input tensor of shape [batch, layer_dims[0]]

        Returns:
            Output tensor of shape [batch, layer_dims[-1]]
        """
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if self.dropout is not None and i < len(self.layers) - 1:
                x = self.dropout(x)
        return x


class DeepOKAN(BaseDeepO):
    """
    DeepOKAN.

    Causal per-timestep prediction uses zero-padded input windows. Inherited
    forward shapes are [B, 1] for per-timestep and [B, 1, sensor_dim] for sequences.
    """

    def __init__(
        self,
        sensor_dim: int = 4000,
        latent_dim: int = 100,
        branch_layers: Optional[List[int]] = None,
        trunk_layers: Optional[List[int]] = None,
        degree: int = 7,  # Chebyshev polynomial degree (T0-T7)
        dropout: float = 0.0,
        init_type: str = "pytorch_default",
    ):
        """
        Initialize DeepOKAN.

        Args:
            sensor_dim: Number of input timesteps (default 4000 for CDON)
            latent_dim: Dimension of latent space (default 100)
            branch_layers: Hidden layer sizes for branch network
                          Default: [83, 83, 83] for ~3M params (shallow 4-layer network)
            trunk_layers: Hidden layer sizes for trunk network
                         Default: [83, 83, 83] for ~3M params (shallow 4-layer network)
            degree: Chebyshev polynomial degree (default 7 for T0-T7)
            dropout: Dropout rate between hidden layers (default 0.0)
            init_type: Initialization type for all layers (branch, trunk, and output). Options:
                      - 'kaiming': Kaiming/He init with std=sqrt(2/fan_in)
                      - 'xavier': Xavier/Glorot init with std=sqrt(2/(fan_in+fan_out))
                      - 'pytorch_default': PyTorch Linear default (Kaiming uniform with a=sqrt(5))
                      Applied consistently across all ChebyshevKAN layers and output layer.
        """
        super().__init__()
        self.init_type = init_type

        self.sensor_dim = sensor_dim
        self.latent_dim = latent_dim
        self.branch_layers = branch_layers or [83, 83, 83]
        self.trunk_layers = trunk_layers or [83, 83, 83]
        self.degree = degree
        self.dropout = dropout

        # [input_dim, *hidden_layers, output_dim]
        branch_dims = [sensor_dim] + self.branch_layers + [latent_dim]
        trunk_dims = [1] + self.trunk_layers + [latent_dim]

        self.branch = _ChebyshevKANNetwork(
            layer_dims=branch_dims, degree=degree, dropout=dropout, init_type=init_type
        )  # [batch, latent_dim]

        self.trunk = _ChebyshevKANNetwork(
            layer_dims=trunk_dims, degree=degree, dropout=dropout, init_type=init_type
        )  # [batch, latent_dim]

        self.output_layer = _ChebyshevKANLayer(
            latent_dim, 1, degree=degree, init_type=init_type
        )

    def _apply_output_layer(
        self, combined: torch.Tensor, is_batched_3d: bool = False
    ) -> torch.Tensor:
        """
        Apply KAN output layer .

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
        if is_batched_3d:
            orig_shape = combined.shape
            combined_flat = combined.reshape(-1, self.latent_dim)
            output_flat = self.output_layer(combined_flat)  # [batch*seq_len, 1]
            output = output_flat.reshape(orig_shape[0], orig_shape[1], 1)
            return output.squeeze(-1).unsqueeze(1)  # [batch, 1, seq_len]
        else:
            return self.output_layer(combined)  # [batch, 1]
