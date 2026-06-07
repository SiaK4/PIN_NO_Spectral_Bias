"""Spectral utilities for BSP loss and evaluation metrics."""

import numpy as np
import torch

from .constants import LX_CDON


def compute_binned_spectral_density(
    signal: torch.Tensor,
    lx: float = LX_CDON,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute the physical binned spectral density used by the BSP loss.

    Full FFT, physical wavenumbers from lx, shell-sum binning (sum / dk),
    channels summed before binning, DC included. Returns centers [n_bins]
    and Ek [B, n_bins]. Computed in float64.
    """
    if device is None:
        device = signal.device
    u = signal.to(device=device, dtype=torch.float64)

    # [B, nx] -> [B, 1, nx]
    if u.ndim == 2:
        u = u.unsqueeze(1)
    if u.ndim != 3:
        raise ValueError(f"Expected [B, C, nx] or [B, nx], got shape {tuple(u.shape)}")

    B, _C, nx = u.shape
    dx = lx / nx

    # Full FFT (positive and negative k); fftfreq returns cycles per sample.
    k = 2 * np.pi * torch.fft.fftfreq(nx, d=dx, device=u.device, dtype=u.dtype)  # [nx]
    k_mag = k.abs()  # Magnitude of wavenumber

    # FFT with orthonormal normalization.
    Uh = torch.fft.fft(u, dim=-1, norm="ortho")  # [B, C, nx]
    energy_k = 0.5 * (Uh.real**2 + Uh.imag**2).sum(dim=1)  # [B, nx]

    # Binning
    kmax = float(np.pi / dx)  # Nyquist wavenumber
    dk = float(2 * np.pi / lx)  # Wavenumber resolution
    edges = torch.arange(0.0, kmax + dk, dk, device=u.device, dtype=u.dtype)
    centers = 0.5 * (edges[1:] + edges[:-1])
    nb = len(edges) - 1

    bin_idx = torch.bucketize(k_mag, edges) - 1
    bin_idx = bin_idx.clamp(0, nb - 1)

    # Shell-sum
    Ek_sum = torch.zeros(B, nb, device=u.device, dtype=energy_k.dtype)
    idx = bin_idx.unsqueeze(0).expand(B, -1)
    Ek_sum.scatter_add_(1, idx, energy_k)

    Ek = Ek_sum / dk

    return centers, Ek


def compute_rfft_energy(signal: torch.Tensor) -> torch.Tensor:
    """Compute real-input FFT energy over nonnegative frequency bins."""
    fft_result = torch.fft.rfft(signal, dim=-1, norm="ortho")
    return 0.5 * torch.abs(fft_result) ** 2
