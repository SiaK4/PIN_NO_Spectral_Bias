"""
Loss factory for training.

Combined loss = L2 field norm + L2 spectral (BSP) norm, where the spectral term
is dropped for the baseline (use_bsp=False).
"""

from typing import TYPE_CHECKING, Any, Dict, Union

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from configs.loss_config import LossConfig

from .constants import EPSILON_SPECTRAL, LX_CDON
from .spectral_utils import compute_binned_spectral_density


def compute_bsp_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    lx: float = LX_CDON,
    reduction: str = "sum",
) -> torch.Tensor:
    """
    Compute BSP loss (normalized log-spectral L2 norm).

    Args:
        pred: [B, C, T] predictions
        target: [B, C, T] ground truth
        lx: Physical domain length (2π for CDON)
        reduction: 'sum' for scalar (global L2 norm), 'none' for per-sample [B]

    Returns:
        Scalar BSP loss if reduction='sum', or [B] per-sample losses if reduction='none'
    """
    _, spec_pred = compute_binned_spectral_density(pred, lx=lx)
    _, spec_true = compute_binned_spectral_density(target, lx=lx)

    bsp_pred = torch.log10(spec_pred + EPSILON_SPECTRAL)
    bsp_true = torch.log10(spec_true + EPSILON_SPECTRAL)

    temp_min = torch.min(bsp_true)
    temp_max = torch.max(bsp_true)

    nbsp_true = (bsp_true - temp_min) / (temp_max - temp_min)
    nbsp_pred = (bsp_pred - temp_min) / (temp_max - temp_min)

    if reduction == "none":
        # Per-sample L2 norm: [B], cast back to input precision
        return torch.norm(nbsp_true - nbsp_pred, p=2, dim=-1).to(pred.dtype)
    else:
        # Global L2 norm: scalar, cast back to input precision
        return torch.norm(nbsp_true - nbsp_pred, p=2).to(pred.dtype)


class CombinedLoss(nn.Module):
    """
    Combined L2 Norm + BSP loss.

    Computes:
        L = L2_field + L2_spectral  (if use_bsp=True)
        L = L2_field                (if use_bsp=False, baseline)

    where:
        L2_field = torch.norm(y_true - y_pred, p=2)
        L2_spectral = torch.norm(nbsp_true - nbsp_pred, p=2)
        nbsp = normalized binned spectral power (log + min-max normalization)
    """

    def __init__(self, use_bsp: bool = False, lx: float = LX_CDON):
        super().__init__()
        self.use_bsp = use_bsp
        self.lx = lx

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute combined L2 field + spectral loss (scalar)."""
        loss1 = torch.norm(target - pred, p=2)
        if not self.use_bsp:
            return loss1
        loss2 = compute_bsp_loss(pred, target, self.lx, reduction="sum")
        return loss1 + loss2


def create_loss(config: Union["LossConfig", Dict[str, Any]]) -> nn.Module:
    """
    Create loss module from config.

    Supports:
    - 'baseline': L2 norm only
    - 'bsp': L2 norm + BSP
    """
    from configs.loss_config import LossConfig

    if isinstance(config, dict):
        config = LossConfig.from_dict(config)

    loss_type = config.loss_type
    params = config.loss_params
    lx = params.get("lx", LX_CDON)

    if loss_type == "baseline":
        return CombinedLoss(use_bsp=False, lx=lx)
    elif loss_type == "bsp":
        return CombinedLoss(use_bsp=True, lx=lx)
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}. Use 'baseline' or 'bsp'.")
