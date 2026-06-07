"""
Configuration dataclasses for neural operator models.

Provides structured configuration for DeepONet, FNO, CNO, and DeepOKAN.
All configs inherit from BaseConfig for serialization support (JSON save/load).
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List

from configs.base_config import BaseConfig
from src.core.evaluation.constants import (
    BRANCH_ACTIVATION_DEFAULT,
    SIREN_W0_DEFAULT,
    TRUNK_ACTIVATION_DEFAULT,
)


@dataclass
class DeepONetConfig(BaseConfig):
    """
    Configuration for DeepONet model.

    Attributes:
        sensor_dim: Input dimension (number of timesteps)
        latent_dim: Dimension of latent space for branch-trunk combination
        branch_layers: Hidden layer sizes for branch network
        trunk_layers: Hidden layer sizes for trunk network
        branch_activation: Activation function for branch network
                          'tanh' is recommended for input encoding
        trunk_activation: Activation function for trunk network
                         'siren' is recommended for coordinate encoding
        dropout: Dropout rate for regularization
        siren_w0: SIREN w0 parameter (initial frequency scale, default 30.0)
        init_type: Weight initialization type for non-SIREN layers
                  'pytorch_default' recommended
    """

    sensor_dim: int = 4000
    latent_dim: int = 430
    branch_layers: List[int] = field(default_factory=lambda: [430, 430, 430])
    trunk_layers: List[int] = field(default_factory=lambda: [430, 430, 430])
    branch_activation: str = BRANCH_ACTIVATION_DEFAULT
    trunk_activation: str = TRUNK_ACTIVATION_DEFAULT
    dropout: float = 0.05
    siren_w0: float = SIREN_W0_DEFAULT
    init_type: str = "pytorch_default"

    def __repr__(self) -> str:
        """String representation showing key parameters."""
        return (
            f"DeepONetConfig(\n"
            f"  sensor_dim={self.sensor_dim},\n"
            f"  latent_dim={self.latent_dim},\n"
            f"  branch_layers={self.branch_layers},\n"
            f"  trunk_layers={self.trunk_layers},\n"
            f"  branch_activation='{self.branch_activation}',\n"
            f"  trunk_activation='{self.trunk_activation}',\n"
            f"  dropout={self.dropout},\n"
            f"  siren_w0={self.siren_w0},\n"
            f"  init_type='{self.init_type}'\n"
            f")"
        )


@dataclass
class FNOConfig(BaseConfig):
    """
    Configuration for FNO model (TFNO is being used).

    Attributes:
        n_modes: Number of Fourier modes to keep (100 for ~3M target)
        hidden_channels: Hidden channel dimension (256 for ~3M target)
        n_layers: Number of TFNO layers
        in_channels: Number of input channels
        out_channels: Number of output channels
        factorization: Tensor factorization type ('tucker', 'cp', 'tt')
        implementation: Implementation type ('factorized', 'reconstructed')
        rank: Factorization rank as fraction (0.10 = 10% of dense params)
    """

    n_modes: int = 100
    hidden_channels: int = 256
    n_layers: int = 4
    in_channels: int = 1
    out_channels: int = 1
    factorization: str = "tucker"
    implementation: str = "factorized"
    rank: float = 0.10

    def __repr__(self) -> str:
        """String representation showing key parameters."""
        return (
            f"FNOConfig(\n"
            f"  n_modes={self.n_modes},\n"
            f"  hidden_channels={self.hidden_channels},\n"
            f"  n_layers={self.n_layers},\n"
            f"  factorization='{self.factorization}',\n"
            f"  rank={self.rank}\n"
            f")"
        )


@dataclass
class CNOConfig(BaseConfig):
    """
    Configuration for CNO model.

    Note that CNO requires input pre-interpolated to internal_size (4096).
    Use CDONDataset with target_signal_length=4096 for CNO training.

    Architecture uses 4x downsampling:
        - Input: internal_size (4096) - data must be pre-interpolated
        - Resolution progression: 4096 → 1024 → 256 → 64 → 16
        - Output: internal_size (4096) - interpolate back to 4000 during evaluation

    Attributes:
        in_channels: Number of input channels
        out_channels: Number of output channels
        n_res: ResBlocks per level (except bottleneck)
        n_res_neck: ResBlocks in bottleneck
        dropout: Dropout rate for regularization (0.0 = no dropout)
    """

    in_channels: int = 1
    out_channels: int = 1
    n_res: int = 1
    n_res_neck: int = 1
    dropout: float = 0.05


@dataclass
class DeepOKANConfig(BaseConfig):
    """
    Configuration for DeepOKAN model.

    Attributes:
        sensor_dim: Input dimension (number of timesteps)
        latent_dim: Dimension of latent space for branch-trunk combination
        branch_layers: Hidden layer sizes for branch network
        trunk_layers: Hidden layer sizes for trunk network
        degree: Chebyshev polynomial degree
        dropout: Dropout rate for regularization
        init_type: Weight initialization type for KAN layers
    """

    sensor_dim: int = 4000
    latent_dim: int = 100
    branch_layers: List[int] = field(default_factory=lambda: [83, 83, 83])
    trunk_layers: List[int] = field(default_factory=lambda: [83, 83, 83])
    degree: int = 7
    dropout: float = 0.05
    init_type: str = "pytorch_default"

    def __repr__(self) -> str:
        """String representation showing key parameters."""
        return (
            f"DeepOKANConfig(\n"
            f"  sensor_dim={self.sensor_dim},\n"
            f"  latent_dim={self.latent_dim},\n"
            f"  branch_layers={self.branch_layers},\n"
            f"  trunk_layers={self.trunk_layers},\n"
            f"  degree={self.degree},\n"
            f"  dropout={self.dropout},\n"
            f"  init_type='{self.init_type}'\n"
            f")"
        )


# Type alias for model config dataclasses.
ModelConfig = DeepONetConfig | DeepOKANConfig | FNOConfig | CNOConfig


# Model config defaults.
# Keep these synchronized with the dataclass defaults above.

MODEL_CONFIGS: Dict[str, Dict[str, Any]] = {
    "deeponet": DeepONetConfig().to_dict(),
    "deepokan": DeepOKANConfig().to_dict(),
    "fno": FNOConfig().to_dict(),
    "cno": CNOConfig().to_dict(),
}
