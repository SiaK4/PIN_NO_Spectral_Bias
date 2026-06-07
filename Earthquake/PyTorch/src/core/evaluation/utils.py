"""
Utility functions for model evaluation.

Provides helper functions for evaluating models, including handling
DeepONet's dual-forward architecture and CNO's resolution post-processing.
"""

import torch
import torch.nn.functional as F
from src.core.evaluation.constants import SIGNAL_LENGTH_CDON


def get_model_predictions(
    model: torch.nn.Module, inputs: torch.Tensor, use_causal: bool = True
) -> torch.Tensor:
    """
    Get model predictions with causal evaluation when requested.

    For DeepONet/DeepOKAN, forward_causal_sequence() evaluates timestep t with
    inputs [0, ..., t], as in per-timestep training. FNO/CNO use forward().

    Args:
        model: Neural operator model (DeepONet, DeepOKAN, FNO, or CNO)
        inputs: Input tensor [B, 1, T] for sequence mode
        use_causal: If True and model has forward_causal_sequence, use it

    Returns:
        predictions: [B, 1, T] tensor of full sequence predictions
    """
    if use_causal and hasattr(model, "forward_causal_sequence"):
        return model.forward_causal_sequence(inputs)
    return model(inputs)


def postprocess_cno_predictions(
    predictions: torch.Tensor, target_size: int = SIGNAL_LENGTH_CDON
) -> torch.Tensor:
    """
    Post-process CNO predictions by interpolating from internal size to target size.

    CNO processes at CNO_INTERNAL_SIZE (4096) internally for efficient U-Net
    downsampling (4096 → 1024 → 256 → 64 → 16). This function interpolates
    predictions back to the original CDON signal length (4000) for evaluation.

    Args:
        predictions: CNO predictions tensor [B, 1, CNO_INTERNAL_SIZE]
        target_size: Target sequence length (default: SIGNAL_LENGTH_CDON = 4000)

    Returns:
        Interpolated predictions [B, 1, target_size]
    """
    current_size = predictions.shape[-1]
    if current_size == target_size:
        return predictions

    return F.interpolate(
        predictions, size=target_size, mode="linear", align_corners=True
    )
