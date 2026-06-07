"""
Loss function configuration for neural operator training.

Loss Formulation
================
Field loss:    L_field = L2 Norm loss (torch.norm(y-ŷ, p=2)
Spectral loss: L_spectral = BSP loss
Combined:      L = L_field + w_bsp * L_spectral
"""

import copy
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from src.core.evaluation.constants import (
    EPSILON,
    LX_CDON,
    SIGNAL_LENGTH_CDON,
    W_BSP,
)


@dataclass
class LossConfig:
    """
    Configuration for loss functions in neural operator training.

    Attributes:
        loss_type: Type of loss function to use
            - 'baseline': L2 Norm loss (torch.norm(y-ŷ, p=2))
            - 'bsp':      L2 Norm + BSP (fixed weights)
        loss_params: Dictionary of loss-specific parameters
        description: Optional description for logging
    """

    loss_type: str
    loss_params: Dict[str, Any] = field(default_factory=dict)
    description: Optional[str] = None

    def __post_init__(self):
        """Validate loss configuration."""
        if self.loss_params is not None:
            self.loss_params = copy.deepcopy(self.loss_params)
        else:
            self.loss_params = {}

        valid_types = ["baseline", "bsp"]
        if self.loss_type not in valid_types:
            raise ValueError(
                f"Invalid loss_type: {self.loss_type}. " f"Must be one of {valid_types}"
            )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "loss_type": self.loss_type,
            "loss_params": copy.deepcopy(self.loss_params),
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "LossConfig":
        """Create LossConfig from dictionary."""
        return cls(
            loss_type=config_dict["loss_type"],
            loss_params=copy.deepcopy(config_dict.get("loss_params", {})),
            description=config_dict.get("description"),
        )

    def __repr__(self) -> str:
        params_str = ", ".join(f"{k}={v}" for k, v in self.loss_params.items())
        desc_str = f" ({self.description})" if self.description else ""
        return f"LossConfig(type={self.loss_type}, params={{{params_str}}}){desc_str}"


# Standard configs

BASELINE_CONFIG = LossConfig(
    loss_type="baseline",
    loss_params={},
    description="Baseline: L2 Norm only",
)

BSP_CONFIG = LossConfig(
    loss_type="bsp",
    loss_params={
        "w_bsp": W_BSP,
        "lx": LX_CDON,
        "epsilon": EPSILON,
        "signal_length": SIGNAL_LENGTH_CDON,
    },
    description="L2 Norm + BSP (fixed weights)",
)

# Loss config map

LOSS_CONFIG_MAP = {
    "baseline": BASELINE_CONFIG,
    "bsp": BSP_CONFIG,
}
