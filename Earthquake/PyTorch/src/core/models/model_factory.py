"""
Model factory for creating models.

Provides centralized model instantiation for dual-batch training:
- DeepONet: Per-timestep field loss + full-sequence BSP loss
- FNO/CNO: Full-sequence data for both field and BSP losses
"""

from typing import Any, Dict, Optional

import torch.nn as nn
from configs.model_configs import CNOConfig, DeepOKANConfig, DeepONetConfig, FNOConfig

from .cno import CNO
from .deepokan import DeepOKAN
from .deeponet import DeepONet
from .fno import FNO

_MODEL_REGISTRY = {
    "deeponet": (DeepONet, DeepONetConfig),
    "deepokan": (DeepOKAN, DeepOKANConfig),
    "fno": (FNO, FNOConfig),
    "cno": (CNO, CNOConfig),
}


def model_supports_per_timestep(model: nn.Module) -> bool:
    """Check if model supports per-timestep forward."""
    return getattr(model, "supports_per_timestep", False)


def create_model(arch: str, config: Optional[Dict[str, Any]] = None) -> nn.Module:
    """
    Factory function to create models.

    Args:
        arch: Model architecture name ('deeponet', 'deepokan', 'fno', 'cno')
        config: Optional configuration dictionary with model hyperparameters.
                If None, uses default hyperparameters.

    Returns:
        Initialized nn.Module of the specified architecture

    Raises:
        ValueError: If arch is not recognized
    """
    arch = arch.lower()
    if arch not in _MODEL_REGISTRY:
        raise ValueError(
            f"Unknown architecture: '{arch}'. Valid options: {list(_MODEL_REGISTRY.keys())}"
        )

    model_cls, config_cls = _MODEL_REGISTRY[arch]
    default_config = config_cls()
    defaults = {
        field: getattr(default_config, field)
        for field in default_config.__dataclass_fields__
    }
    merged = {**defaults, **(config or {})}
    return model_cls(**merged)
