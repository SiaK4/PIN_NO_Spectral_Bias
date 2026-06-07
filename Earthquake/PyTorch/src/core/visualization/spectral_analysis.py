"""
Spectral bias analysis.

Spectrum functions expect denormalized physical-space data.
"""

from typing import Tuple

import numpy as np
import torch
from configs.visualization_config import SAMPLING_RATE_HZ


def normalized_freq_to_hz(freq_normalized: np.ndarray) -> np.ndarray:
    """
    Convert normalized frequency [0, 0.5] to Hz [0, 25].

    For CDON dataset:
    - Sampling rate: 50 Hz
    - Nyquist frequency: 25 Hz (at normalized freq = 0.5)

    Args:
        freq_normalized: Normalized frequency array [0, 0.5]

    Returns:
        freq_hz: Frequency in Hz [0, 25]

    Example:
        >>> freq_norm = np.array([0.0, 0.25, 0.5])
        >>> freq_hz = normalized_freq_to_hz(freq_norm)
        >>> print(freq_hz)  # [0.0, 12.5, 25.0]
    """
    return freq_normalized * SAMPLING_RATE_HZ


def compute_unbinned_spectrum(
    signal: torch.Tensor,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute unbinned frequency spectrum with percentile bands.

    Input: [B, C, T]. Returns frequencies [n_freq] plus median, p16, and p84
    power spectra [n_freq], with the DC component removed.
    """
    if isinstance(signal, torch.Tensor):
        signal_np = signal.cpu().numpy()
    else:
        signal_np = signal

    timesteps = signal_np.shape[-1]

    fft_result = np.fft.rfft(signal_np, axis=-1, norm="ortho")  # [B, C, freq] complex

    power_spectrum_per_sample = 0.5 * np.abs(fft_result) ** 2  # [B, C, freq] real

    power_per_sample = power_spectrum_per_sample.mean(axis=1)

    frequencies = np.fft.rfftfreq(timesteps)
    frequencies_no_dc = frequencies[1:]
    power_no_dc = power_per_sample[:, 1:]  # [B, freq-1]

    energy_median = np.percentile(power_no_dc, 50, axis=0)  # Median
    energy_p16 = np.percentile(power_no_dc, 16, axis=0)  # Lower bound ≈ -1σ
    energy_p84 = np.percentile(power_no_dc, 84, axis=0)  # Upper bound ≈ +1σ

    return frequencies_no_dc, energy_median, energy_p16, energy_p84


def compute_temporal_gradient(signal: np.ndarray, dt: float = 0.02) -> np.ndarray:
    """
    Compute du/dt using central finite differences.

    Args:
        signal: Array [T] or batched [B, T] / [B, C, T]
        dt: Time step in seconds (default: 0.02 for CDON 50Hz sampling)

    Returns:
        gradient: Same shape as input, du/dt
    """
    orig_shape = signal.shape
    orig_ndim = signal.ndim

    if signal.ndim == 1:
        signal = signal[np.newaxis, :]
    elif signal.ndim == 3:
        B, C, T = signal.shape
        signal = signal.reshape(B * C, T)

    gradient = np.zeros_like(signal)

    # Central differences for interior points
    gradient[:, 1:-1] = (signal[:, 2:] - signal[:, :-2]) / (2 * dt)

    # Forward difference at left boundary
    gradient[:, 0] = (signal[:, 1] - signal[:, 0]) / dt

    # Backward difference at right boundary
    gradient[:, -1] = (signal[:, -1] - signal[:, -2]) / dt

    if orig_ndim == 1:
        return gradient.squeeze(0)
    return gradient.reshape(orig_shape)


def compute_temporal_laplacian(signal: np.ndarray, dt: float = 0.02) -> np.ndarray:
    """
    Compute d²u/dt² using central finite differences.

    Args:
        signal: Array [T] or batched [B, T] / [B, C, T]
        dt: Time step in seconds (default: 0.02 for CDON 50Hz sampling)

    Returns:
        laplacian: Same shape as input, d²u/dt²
    """
    orig_shape = signal.shape
    orig_ndim = signal.ndim

    if signal.ndim == 1:
        signal = signal[np.newaxis, :]
    elif signal.ndim == 3:
        B, C, T = signal.shape
        signal = signal.reshape(B * C, T)

    laplacian = np.zeros_like(signal)
    dt_sq = dt**2

    # Central differences for interior points
    laplacian[:, 1:-1] = (signal[:, 2:] - 2 * signal[:, 1:-1] + signal[:, :-2]) / dt_sq

    # Second-order forward difference at left boundary
    # u''[0] ≈ (u[2] - 2*u[1] + u[0]) / dt² (using points 0,1,2)
    laplacian[:, 0] = (signal[:, 2] - 2 * signal[:, 1] + signal[:, 0]) / dt_sq

    # Second-order backward difference at right boundary
    # u''[-1] ≈ (u[-1] - 2*u[-2] + u[-3]) / dt² (using points -3,-2,-1)
    laplacian[:, -1] = (signal[:, -1] - 2 * signal[:, -2] + signal[:, -3]) / dt_sq

    if orig_ndim == 1:
        return laplacian.squeeze(0)
    return laplacian.reshape(orig_shape)


def compute_temporal_gradient_interpolated(
    signal: np.ndarray, dt: float = 0.02, upsample_factor: int = 4
) -> np.ndarray:
    """
    Compute du/dt with upsampling for numerical stability.

    Upsamples 4000→16000 timesteps before finite differences,
    then downsamples result to original length.

    Args:
        signal: Array [T] or batched [B, T] / [B, C, T]
        dt: Time step in seconds (default: 0.02 for CDON 50Hz sampling)
        upsample_factor: Factor to upsample before differentiation (default: 4)

    Returns:
        gradient: Same shape as input, du/dt
    """
    import torch
    import torch.nn.functional as F

    signal_t = torch.from_numpy(signal).float()
    orig_shape = signal_t.shape
    if signal_t.ndim == 1:
        signal_t = signal_t.unsqueeze(0).unsqueeze(0)
    elif signal_t.ndim == 2:
        signal_t = signal_t.unsqueeze(1)

    T = signal_t.shape[-1]
    T_up = T * upsample_factor
    dt_up = dt / upsample_factor

    signal_up = F.interpolate(signal_t, size=T_up, mode="linear", align_corners=True)
    signal_up_np = signal_up.squeeze(1).numpy()
    grad_up = compute_temporal_gradient(signal_up_np, dt=dt_up)

    grad_up_t = torch.from_numpy(grad_up).float().unsqueeze(1)
    grad_down = F.interpolate(grad_up_t, size=T, mode="linear", align_corners=True)

    result = grad_down.squeeze(1).numpy()
    if len(orig_shape) == 1:
        result = result.squeeze(0)
    return result


def compute_temporal_laplacian_interpolated(
    signal: np.ndarray, dt: float = 0.02, upsample_factor: int = 4
) -> np.ndarray:
    """
    Compute d²u/dt² with upsampling for numerical stability.

    Second derivatives amplify errors quadratically - upsampling helps.
    Upsamples 4000→16000 timesteps before finite differences,
    then downsamples result to original length.

    Args:
        signal: Array [T] or batched [B, T] / [B, C, T]
        dt: Time step in seconds (default: 0.02 for CDON 50Hz sampling)
        upsample_factor: Factor to upsample before differentiation (default: 4)

    Returns:
        laplacian: Same shape as input, d²u/dt²
    """
    import torch
    import torch.nn.functional as F

    signal_t = torch.from_numpy(signal).float()
    orig_shape = signal_t.shape
    if signal_t.ndim == 1:
        signal_t = signal_t.unsqueeze(0).unsqueeze(0)
    elif signal_t.ndim == 2:
        signal_t = signal_t.unsqueeze(1)

    T = signal_t.shape[-1]
    T_up = T * upsample_factor
    dt_up = dt / upsample_factor

    signal_up = F.interpolate(signal_t, size=T_up, mode="linear", align_corners=True)
    signal_up_np = signal_up.squeeze(1).numpy()
    lap_up = compute_temporal_laplacian(signal_up_np, dt=dt_up)

    lap_up_t = torch.from_numpy(lap_up).float().unsqueeze(1)
    lap_down = F.interpolate(lap_up_t, size=T, mode="linear", align_corners=True)

    result = lap_down.squeeze(1).numpy()
    if len(orig_shape) == 1:
        result = result.squeeze(0)
    return result
