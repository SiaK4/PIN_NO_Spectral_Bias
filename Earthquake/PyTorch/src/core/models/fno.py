"""
Tensorized Fourier Neural Operator (TFNO).

Wrapper around neuralop's TFNO.

TFNO uses Tucker factorization for parameter efficiency. Current neuralop
versions may store spectral parameters as complex tensors, so SOAP requires
complex-gradient support. Alternatively you can downgrade neuralop.
"""

import torch
import torch.nn as nn

try:
    from neuralop.models import TFNO

    NEURALOP_AVAILABLE = True
except ImportError:
    NEURALOP_AVAILABLE = False
    TFNO = None


class FNO(nn.Module):
    """TFNO wrapper."""

    supports_per_timestep = False

    def __init__(
        self,
        n_modes: int,
        hidden_channels: int,
        n_layers: int,
        in_channels: int,
        out_channels: int,
        factorization: str,
        implementation: str,
        rank: float,
    ):
        """
        Initialize TFNO.

        Defaults are set in configs/model_configs.py.

        Args:
            n_modes: Number of Fourier modes to keep
            hidden_channels: Hidden channel dimension
            n_layers: Number of TFNO layers
            in_channels: Number of input channels
            out_channels: Number of output channels
            factorization: Tensor factorization type ('tucker', 'cp', 'tt')
            implementation: Implementation type ('factorized', 'reconstructed')
            rank: Factorization rank as fraction of full rank

        Raises:
            ImportError: If neuralop is not installed
        """
        super().__init__()

        if not NEURALOP_AVAILABLE:
            raise ImportError(
                "neuralop is required for FNO (TFNO). "
                "Install with: pip install -U neuraloperator"
            )

        self.n_modes = n_modes
        self.hidden_channels = hidden_channels
        self.n_layers = n_layers
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.factorization = factorization
        self.implementation = implementation
        self.rank = rank

        self.fno = TFNO(
            n_modes=(self.n_modes,),
            hidden_channels=self.hidden_channels,
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            n_layers=self.n_layers,
            factorization=self.factorization,
            implementation=self.implementation,
            rank=self.rank,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through TFNO.

        Args:
            x: Input tensor of shape [batch, channels, timesteps]

        Returns:
            Output tensor of shape [batch, channels, timesteps]
        """
        return self.fno(x)
