"""
Evaluation metrics for predictions (field nRMSE, log-spectral nRMSE,
Barron norm error).

All metrics use dataset-wide min/max statistics computed on test data.
This keeps comparisons consistent across models trained with different
normalization strategies.
"""

import warnings
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import torch

from .constants import EPSILON, EPSILON_SPECTRAL, GLOBAL_NRMSE_STATS
from .spectral_utils import compute_rfft_energy


@dataclass
class EvaluationResults:
    """Container for evaluation results with statistics.

    All metrics use per-earthquake normalization for comparison
    between models trained with different normalization strategies.
    """

    field_nrmse_mean: float
    field_nrmse_std: float
    log_spectral_nrmse_mean: float
    log_spectral_nrmse_std: float
    barron_norm_error_mean: float
    barron_norm_error_std: float
    n_samples: int
    # Per-quintile log spectral errors
    log_spectral_quintile_errors: Dict[str, float] | None = None

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        result = {
            "field_nrmse_mean": self.field_nrmse_mean,
            "field_nrmse_std": self.field_nrmse_std,
            "log_spectral_nrmse_mean": self.log_spectral_nrmse_mean,
            "log_spectral_nrmse_std": self.log_spectral_nrmse_std,
            "barron_norm_error_mean": self.barron_norm_error_mean,
            "barron_norm_error_std": self.barron_norm_error_std,
            "n_samples": self.n_samples,
        }
        if self.log_spectral_quintile_errors is not None:
            result.update(self.log_spectral_quintile_errors)
        return result


def compute_nrmse(
    pred: torch.Tensor,
    target: torch.Tensor,
    kind: str = "minmax",
    epsilon: float = EPSILON,
    global_stats: Optional[Dict[str, float]] = None,
) -> torch.Tensor:
    """Compute per-sample nRMSE [B] with global min-max statistics."""
    if pred.shape != target.shape:
        raise ValueError(
            f"Shape mismatch: pred {pred.shape} vs target {target.shape}. "
            f"Predictions and targets must have identical shapes."
        )

    if kind != "minmax":
        raise ValueError(f"Only 'minmax' kind supported. Got: {kind}")

    if global_stats is None:
        raise ValueError(
            "global_stats must be provided. Use the appropriate metric-specific "
            "function (compute_field_nrmse, compute_log_spectral_nrmse, etc.) "
            "which will pass the correct global stats."
        )

    pred = pred.double()
    target = target.double()

    pred_flat = pred.view(pred.shape[0], -1)
    target_flat = target.view(target.shape[0], -1)

    shift = global_stats["global_min"]
    scale = global_stats["global_max"] - shift + epsilon

    target_norm = (target_flat - shift) / scale
    pred_norm = (pred_flat - shift) / scale

    mse = torch.mean((pred_norm - target_norm) ** 2, dim=-1)  # [B]
    nrmse = torch.sqrt(mse)

    if torch.isnan(nrmse).any():
        warnings.warn(
            "NaN values detected in nRMSE computation. "
            "This may indicate numerical instability or invalid inputs.",
            RuntimeWarning,
        )

    return nrmse.float()


def compute_field_nrmse(
    pred: torch.Tensor, target: torch.Tensor, epsilon: float = EPSILON
) -> torch.Tensor:
    """Compute field nRMSE [B] using global min-max normalization."""
    if GLOBAL_NRMSE_STATS is None:
        raise RuntimeError(
            "Global nRMSE stats not found. Precompute them before evaluation."
        )
    return compute_nrmse(
        pred, target, epsilon=epsilon, global_stats=GLOBAL_NRMSE_STATS["field"]
    )


def compute_energy_spectrum(signal: torch.Tensor) -> torch.Tensor:
    """
    Compute raw real-input FFT energy over nonnegative frequency bins.

    The energy spectrum is defined as $E(k) = 0.5 * |FFT(x)|^2$ where k is the frequency index.
    The 0.5 factor follows the BSP paper convention for energy per mode.

    Computation is performed at float64 precision for numerical stability.
    Returns float64 so downstream log/sqrt operations stay in float64.

    Args:
        signal: Time series [batch, channels, timesteps] or [batch, timesteps]
    Returns:
        Energy spectrum [batch, n_freqs] where n_freqs = timesteps // 2 + 1, dtype=float64
    """
    signal = signal.double()

    if signal.ndim == 3:
        signal = signal.mean(dim=1)  # [B, T]

    energy = compute_rfft_energy(signal)  # [B, T//2 + 1]

    return energy


def compute_barron_spectrum(signal: torch.Tensor, alpha: float = 0.0) -> torch.Tensor:
    """
    Compute binned Barron spectrum [B, nbins-1].

    Formula: weighted_energy = sqrt(E_k) * |k|^(1+alpha), binned by |k|.
    """
    signal = signal.double()

    if signal.ndim == 2:
        signal = signal.unsqueeze(1)  # [B, T] -> [B, 1, T]

    B, _C, N = signal.shape
    device = signal.device
    dtype = torch.float64  # Use float64 for all intermediate computations

    # Full FFT along the time axis, with orthonormal scaling.
    fft_result = torch.fft.fft(signal, dim=-1, norm="ortho")  # [B, C, N]

    # Energy per frequency, summed over channels.
    energy_per_channel = 0.5 * (fft_result.real**2 + fft_result.imag**2)  # [B, C, N]
    energy = energy_per_channel.sum(dim=1)  # [B, N] - sum over channels

    # Wavenumber grid using unit spacing.
    L = N  # Domain length = N for unit spacing
    k = 2 * torch.pi * torch.fft.fftfreq(N, d=L / N, device=device)  # [N]
    k_mag = torch.abs(k)  # [N]

    kmax = 2 * torch.pi * (N / 2) / L
    kmin = 2 * torch.pi / L
    nbins = N // 2 + 1

    wave_numbers = torch.linspace(kmin, kmax, nbins, device=device, dtype=dtype)

    # Use the spectral epsilon for float64 spectral computations.
    weighted_energy = torch.sqrt(energy + EPSILON_SPECTRAL) * torch.pow(
        k_mag + EPSILON_SPECTRAL, 1 + alpha
    )  # [B, N]

    barron_spectrum = torch.zeros(B, nbins - 1, device=device, dtype=dtype)

    for bin_idx in range(nbins - 1):
        lo = wave_numbers[bin_idx]
        hi = wave_numbers[bin_idx + 1]
        mask = (k_mag >= lo) & (k_mag < hi)  # [N]
        if mask.any():
            barron_spectrum[:, bin_idx] = weighted_energy[:, mask].sum(dim=1)

    # Scale by the number of bins.
    barron_spectrum = barron_spectrum * nbins

    return barron_spectrum  # [B, nbins-1]


def compute_barron_norm_error(
    pred: torch.Tensor,
    target: torch.Tensor,
    alpha: float = 0.0,
    epsilon: float = EPSILON,
) -> torch.Tensor:
    """
    Compute Barron spectrum nRMSE [B].

    BN_pred and BN_target are normalized with global min-max statistics, then
    compared by sqrt(mean((norm_pred - norm_target)^2)).
    """
    if GLOBAL_NRMSE_STATS is None:
        raise RuntimeError(
            "Global nRMSE stats not found. Precompute them before evaluation."
        )

    if pred.shape != target.shape:
        raise ValueError(
            f"Shape mismatch: pred {pred.shape} vs target {target.shape}. "
            f"Predictions and targets must have identical shapes."
        )

    bn_pred = compute_barron_spectrum(pred, alpha)  # [B, nbins-1]
    bn_target = compute_barron_spectrum(target, alpha)  # [B, nbins-1]

    return compute_nrmse(
        bn_pred,
        bn_target,
        epsilon=epsilon,
        global_stats=GLOBAL_NRMSE_STATS["barron_spectrum"],
    )


def compute_log_spectral_nrmse(
    pred: torch.Tensor, target: torch.Tensor, epsilon: float = EPSILON
) -> torch.Tensor:
    """Compute log-spectrum nRMSE with global min-max statistics."""
    if GLOBAL_NRMSE_STATS is None:
        raise RuntimeError(
            "Global nRMSE stats not found. Precompute them before evaluation."
        )

    if pred.shape != target.shape:
        raise ValueError(
            f"Shape mismatch: pred {pred.shape} vs target {target.shape}. "
            f"Predictions and targets must have identical shapes."
        )

    pred_ek = compute_energy_spectrum(pred)  # [B, n_freqs], float64
    target_ek = compute_energy_spectrum(target)  # [B, n_freqs], float64

    log_pred = torch.log(pred_ek + EPSILON_SPECTRAL)  # [B, n_freqs]
    log_target = torch.log(target_ek + EPSILON_SPECTRAL)  # [B, n_freqs]

    return compute_nrmse(
        log_pred,
        log_target,
        epsilon=epsilon,
        global_stats=GLOBAL_NRMSE_STATS["log_spectrum"],
    )


def compute_log_spectral_quintile_errors(
    pred: torch.Tensor, target: torch.Tensor, n_quintiles: int = 5
) -> Dict[str, float]:
    """Compute mean/std log spectral RMSE for each frequency quintile."""
    if pred.shape != target.shape:
        raise ValueError(
            f"Shape mismatch: pred {pred.shape} vs target {target.shape}. "
            f"Predictions and targets must have identical shapes."
        )

    pred_ek = compute_energy_spectrum(pred)  # [B, n_freqs], float64
    target_ek = compute_energy_spectrum(target)  # [B, n_freqs], float64

    log_pred = torch.log(pred_ek + EPSILON_SPECTRAL)  # [B, n_freqs]
    log_target = torch.log(target_ek + EPSILON_SPECTRAL)  # [B, n_freqs]

    n_freqs = log_pred.shape[1]
    quintile_size = n_freqs // n_quintiles

    results = {}
    for q in range(n_quintiles):
        start_idx = q * quintile_size
        # Last quintile takes remaining frequencies
        end_idx = (q + 1) * quintile_size if q < n_quintiles - 1 else n_freqs

        log_pred_q = log_pred[:, start_idx:end_idx]  # [B, quintile_size]
        log_target_q = log_target[:, start_idx:end_idx]  # [B, quintile_size]

        mse_q = torch.mean((log_pred_q - log_target_q) ** 2, dim=-1)  # [B]
        rmse_q = torch.sqrt(mse_q)  # [B]

        q_name = f"q{q + 1}"  # q1, q2, ..., q5
        results[f"log_spec_{q_name}_mean"] = rmse_q.mean().item()
        results[f"log_spec_{q_name}_std"] = (
            rmse_q.std().item() if rmse_q.numel() > 1 else 0.0
        )

    return results


def evaluate_model(
    pred: torch.Tensor, target: torch.Tensor, epsilon: float = EPSILON
) -> EvaluationResults:
    """
    Compute all evaluation metrics with statistics.

    All metrics use dataset-wide min/max normalization for consistent comparison
    between models trained with different strategies.

    Args:
        pred: Predicted output [batch, channels, timesteps]
        target: Ground truth [batch, channels, timesteps]
        epsilon: Small constant for numerical stability

    Returns:
        EvaluationResults with mean ± std for each metric
    """
    field_nrmse = compute_field_nrmse(pred, target, epsilon=epsilon)
    log_spec_nrmse = compute_log_spectral_nrmse(pred, target, epsilon=epsilon)
    barron_err = compute_barron_norm_error(pred, target, alpha=0.0, epsilon=epsilon)

    quintile_errors = compute_log_spectral_quintile_errors(pred, target, n_quintiles=5)

    n_samples = field_nrmse.numel()
    if n_samples == 1:  # Necessary to avoid NaN issues
        field_nrmse_std = 0.0
        log_spec_nrmse_std = 0.0
        barron_err_std = 0.0
    else:
        field_nrmse_std = field_nrmse.std().item()
        log_spec_nrmse_std = log_spec_nrmse.std().item()
        barron_err_std = barron_err.std().item()

    return EvaluationResults(
        field_nrmse_mean=field_nrmse.mean().item(),
        field_nrmse_std=field_nrmse_std,
        log_spectral_nrmse_mean=log_spec_nrmse.mean().item(),
        log_spectral_nrmse_std=log_spec_nrmse_std,
        barron_norm_error_mean=barron_err.mean().item(),
        barron_norm_error_std=barron_err_std,
        n_samples=n_samples,
        log_spectral_quintile_errors=quintile_errors,
    )


def compute_velocity_nrmse(
    pred: torch.Tensor, target: torch.Tensor, dt: float = 0.02, epsilon: float = EPSILON
) -> torch.Tensor:
    """
    Compute nRMSE on velocity (du/dt).

    Uses dataset-wide min/max normalization for consistent metrics.

    Args:
        pred: Predicted output [batch, channels, timesteps] or [batch, timesteps]
        target: Ground truth (same shape as pred)
        dt: Time step in seconds (0.02s for 50Hz)
        epsilon: Small constant for numerical stability

    Returns:
        torch.Tensor: Per-sample velocity nRMSE, shape [batch], dtype=float32
    """
    from src.core.visualization.spectral_analysis import (
        compute_temporal_gradient_interpolated,
    )

    if GLOBAL_NRMSE_STATS is None:
        raise RuntimeError(
            "Global nRMSE stats not found. Precompute them before evaluation."
        )

    if pred.shape != target.shape:
        raise ValueError(
            f"Shape mismatch: pred {pred.shape} vs target {target.shape}. "
            f"Predictions and targets must have identical shapes."
        )

    if pred.ndim == 3:
        pred = pred[:, 0, :]
        target = target[:, 0, :]

    pred_np = pred.cpu().numpy()
    target_np = target.cpu().numpy()

    pred_grad = compute_temporal_gradient_interpolated(pred_np, dt=dt)
    target_grad = compute_temporal_gradient_interpolated(target_np, dt=dt)

    pred_grad = torch.from_numpy(pred_grad).float()
    target_grad = torch.from_numpy(target_grad).float()

    return compute_nrmse(
        pred_grad,
        target_grad,
        epsilon=epsilon,
        global_stats=GLOBAL_NRMSE_STATS["velocity"],
    )


def compute_acceleration_nrmse(
    pred: torch.Tensor, target: torch.Tensor, dt: float = 0.02, epsilon: float = EPSILON
) -> torch.Tensor:
    """
    Compute nRMSE on acceleration (d²u/dt²).

    Uses dataset-wide min/max normalization for consistent metrics.

    Args:
        pred: Predicted output [batch, channels, timesteps] or [batch, timesteps]
        target: Ground truth (same shape as pred)
        dt: Time step in seconds (0.02s for 50Hz)
        epsilon: Small constant for numerical stability

    Returns:
        torch.Tensor: Per-sample acceleration nRMSE, shape [batch], dtype=float32
    """
    from src.core.visualization.spectral_analysis import (
        compute_temporal_laplacian_interpolated,
    )

    if GLOBAL_NRMSE_STATS is None:
        raise RuntimeError(
            "Global nRMSE stats not found. Precompute them before evaluation."
        )

    if pred.shape != target.shape:
        raise ValueError(
            f"Shape mismatch: pred {pred.shape} vs target {target.shape}. "
            f"Predictions and targets must have identical shapes."
        )

    if pred.ndim == 3:
        pred = pred[:, 0, :]
        target = target[:, 0, :]

    pred_np = pred.cpu().numpy()
    target_np = target.cpu().numpy()

    pred_lap = compute_temporal_laplacian_interpolated(pred_np, dt=dt)
    target_lap = compute_temporal_laplacian_interpolated(target_np, dt=dt)

    pred_lap = torch.from_numpy(pred_lap).float()
    target_lap = torch.from_numpy(target_lap).float()

    return compute_nrmse(
        pred_lap,
        target_lap,
        epsilon=epsilon,
        global_stats=GLOBAL_NRMSE_STATS["acceleration"],
    )


def compute_derivative_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    dt: float = 0.02,
    epsilon: float = EPSILON,
) -> Dict[str, float]:
    """Compute velocity and acceleration nRMSE metrics for [N, C, T] or [N, T]."""
    pred_tensor = torch.from_numpy(predictions).float()
    target_tensor = torch.from_numpy(targets).float()

    velocity_nrmse = compute_velocity_nrmse(
        pred_tensor, target_tensor, dt=dt, epsilon=epsilon
    )
    acceleration_nrmse = compute_acceleration_nrmse(
        pred_tensor, target_tensor, dt=dt, epsilon=epsilon
    )

    n_samples = velocity_nrmse.numel()
    if n_samples == 1:  # Necessary to avoid NaN issues
        velocity_std = 0.0
        acceleration_std = 0.0
    else:
        velocity_std = velocity_nrmse.std().item()
        acceleration_std = acceleration_nrmse.std().item()

    return {
        "velocity_nrmse_mean": velocity_nrmse.mean().item(),
        "velocity_nrmse_std": velocity_std,
        "acceleration_nrmse_mean": acceleration_nrmse.mean().item(),
        "acceleration_nrmse_std": acceleration_std,
    }
