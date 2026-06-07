"""
Shared plotting utilities for evaluation scripts.

Functions follow consistent styling; colors and labels can be customized via
optional dictionaries. Key entry points: compute_fft_spectrum,
plot_4panel_comparison, plot_energy_spectrum_comparison, plot_training_curves.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
from configs.visualization_config import NYQUIST_FREQ_HZ
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from src.core.evaluation.constants import GLOBAL_NRMSE_STATS
from src.core.evaluation.metrics import compute_field_nrmse

from .spectral_analysis import compute_unbinned_spectrum, normalized_freq_to_hz

# Default color schemes

# Colors for loss variants
LOSS_COLORS_DEFAULT = {
    'baseline': '#1f77b4',       # Blue
    'bsp': '#ff7f0e',            # Orange
}

# Labels for loss variants
LOSS_LABELS_DEFAULT = {
    'baseline': 'Baseline',
    'bsp': 'BSP',
}

# Colors for activation ablation
ACTIVATION_COLORS = {
    'siren': '#1f77b4',   # Blue
    'tanh': '#ff7f0e',    # Orange
    'gelu': '#2ca02c',    # Green
    'sin': '#d62728',     # Red
    'relu': '#9467bd',    # Purple
    'requ': '#8c564b',    # Brown
}

# Labels for activation ablation
ACTIVATION_LABELS = {
    'siren': 'SIREN',
    'tanh': 'Tanh',
    'gelu': 'GELU',
    'sin': 'Sin',
    'relu': 'ReLU',
    'requ': 'ReQU',
}

# Colors for optimizer ablation
OPTIMIZER_COLORS = {
    'soap': '#1f77b4',   # Blue
    'adam': '#d62728',   # Red
}

# Labels for optimizer ablation
OPTIMIZER_LABELS = {
    'soap': 'SOAP',
    'adam': 'Adam',
}

# Colors for architecture comparison
ARCH_COLORS = {
    'deeponet': '#1f77b4',   # Blue
    'deepokan': '#ff7f0e',   # Orange
    'fno': '#2ca02c',        # Green
    'cno': '#d62728',        # Red
}

# Labels for architectures
ARCH_LABELS = {
    'deeponet': 'DeepONet',
    'deepokan': 'DeepOKAN',
    'fno': 'FNO',
    'cno': 'CNO',
}


# Sample selection utilities

def select_visualization_samples(
    predictions: np.ndarray,
    targets: np.ndarray,
    n_samples: int = 2
) -> Tuple[list, list]:
    """
    Select samples for visualization based on Field nRMSE extremes.

    Chooses samples with lowest/highest (and optionally median) Field nRMSE
    using the same global min/max normalization as the summary table metrics, to
    show best-/worst-case performance. Sample 0 (the zero sample for the
    homogeneity constraint) is excluded so only real earthquakes are shown.

    Returns (sample_indices, sample_labels) where labels are e.g.
    'Best nRMSE (0.0007)'. predictions/targets are [N, ...] or [N, C, T].
    """
    N = predictions.shape[0]

    if GLOBAL_NRMSE_STATS is None:
        raise RuntimeError(
            "Global nRMSE stats not found. Precompute them before plotting."
        )

    pred_tensor = torch.from_numpy(predictions).float()
    target_tensor = torch.from_numpy(targets).float()

    nrmse_per_sample = compute_field_nrmse(pred_tensor, target_tensor).numpy()  # [N]

    valid_indices = np.where(np.isfinite(nrmse_per_sample))[0]
    valid_indices = valid_indices[valid_indices != 0]  # Exclude zero sample
    if len(valid_indices) == 0:
        return list(range(1, min(n_samples + 1, N))), ['Sample 1', 'Sample 2'][:n_samples]

    sorted_by_error = valid_indices[np.argsort(nrmse_per_sample[valid_indices])]

    sample_indices = []
    sample_labels = []

    if n_samples >= 1:
        best_idx = sorted_by_error[0]
        sample_indices.append(int(best_idx))
        sample_labels.append(f'Best nRMSE ({nrmse_per_sample[best_idx]:.4f})')

    if n_samples >= 2:
        worst_idx = sorted_by_error[-1]
        sample_indices.append(int(worst_idx))
        sample_labels.append(f'Worst nRMSE ({nrmse_per_sample[worst_idx]:.4f})')

    if n_samples >= 3:
        mid_idx = sorted_by_error[len(sorted_by_error) // 2]
        sample_indices.append(int(mid_idx))
        sample_labels.append(f'Median nRMSE ({nrmse_per_sample[mid_idx]:.4f})')

    return sample_indices[:n_samples], sample_labels[:n_samples]



def compute_fft_spectrum(
    signal: np.ndarray,
    dt: float = 0.02
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute rFFT energy spectrum E = 0.5 * |FFT|^2 for signal [T]."""
    n = len(signal)
    fft = np.fft.rfft(signal, norm='ortho')
    freqs = np.fft.rfftfreq(n, dt)
    energy = 0.5 * np.abs(fft) ** 2
    return freqs, energy


def _extract_sample(arr: np.ndarray, idx: int) -> np.ndarray:
    """Extract and flatten a single sample from batched array."""
    if arr.ndim == 3:
        return arr[idx, 0, :]
    elif arr.ndim == 2:
        return arr[idx, :]
    return arr


def plot_4panel_comparison(
    predictions: Dict[str, np.ndarray],
    target: np.ndarray,
    input_signal: np.ndarray,
    sample_idx: int = 0,
    output_path: Optional[str] = None,
    title: Optional[str] = None,
    colors: Optional[Dict[str, str]] = None,
    labels: Optional[Dict[str, str]] = None,
    dt: float = 0.02,
    figsize: Tuple[int, int] = (16, 12),
) -> Tuple[Figure, np.ndarray]:
    """Generate a 2x2 prediction comparison for one sample."""
    colors = colors or LOSS_COLORS_DEFAULT
    labels = labels or LOSS_LABELS_DEFAULT

    target_1d = _extract_sample(target, sample_idx)
    input_1d = _extract_sample(input_signal, sample_idx)

    n_timesteps = len(target_1d)
    t = np.arange(n_timesteps) * dt

    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # Panel 1: FFT Energy Spectrum
    ax = axes[0, 0]
    freqs_gt, energy_gt = compute_fft_spectrum(target_1d, dt)
    ax.semilogy(freqs_gt, energy_gt, 'k-', linewidth=2.5, label='Ground Truth', alpha=0.9)

    for variant_name, pred in predictions.items():
        pred_1d = _extract_sample(pred, sample_idx)
        freqs, energy = compute_fft_spectrum(pred_1d, dt)
        color = colors.get(variant_name, 'gray')
        label = labels.get(variant_name, variant_name)
        ax.semilogy(freqs, energy, color=color, linewidth=1.5, label=label, alpha=0.7)

    ax.set_xlabel('Frequency (Hz)', fontsize=12)
    ax.set_ylabel('Energy', fontsize=12)
    ax.set_title('FFT Energy Spectrum', fontsize=13, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    ax.set_xlim((0, NYQUIST_FREQ_HZ))
    ax.grid(True, alpha=0.3)

    # Panel 2: Time-domain Response
    ax = axes[0, 1]
    ax.plot(t, target_1d, 'k-', linewidth=2, label='Ground Truth', alpha=0.9)

    for variant_name, pred in predictions.items():
        pred_1d = _extract_sample(pred, sample_idx)
        color = colors.get(variant_name, 'gray')
        label = labels.get(variant_name, variant_name)
        ax.plot(t, pred_1d, color=color, linewidth=1, label=label, alpha=0.7)

    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Displacement (m)', fontsize=12)
    ax.set_title('Time-domain Response', fontsize=13, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel 3: Input Signal
    ax = axes[1, 0]
    ax.plot(t, input_1d, 'b-', linewidth=1)
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Acceleration (m/s²)', fontsize=12)
    ax.set_title('Input Acceleration', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Panel 4: Normalized Error
    ax = axes[1, 1]
    target_norm = np.linalg.norm(target_1d)

    for variant_name, pred in predictions.items():
        pred_1d = _extract_sample(pred, sample_idx)
        error = np.abs(pred_1d - target_1d) / (target_norm + 1e-8)
        color = colors.get(variant_name, 'gray')
        label = labels.get(variant_name, variant_name)
        ax.plot(t, error, color=color, linewidth=1, label=label, alpha=0.7)

    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Normalized Error', fontsize=12)
    ax.set_title('Normalized Error', fontsize=13, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)

    # Title and layout
    plot_title = title or f'4-Panel Comparison (Sample {sample_idx})'
    fig.suptitle(plot_title, fontsize=15, fontweight='bold')
    plt.tight_layout()

    # Save if requested
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

    return fig, axes


def plot_energy_spectrum_comparison(
    predictions: Dict[str, Union[np.ndarray, torch.Tensor]],
    targets: Union[np.ndarray, torch.Tensor],
    output_path: Optional[str] = None,
    title: Optional[str] = None,
    colors: Optional[Dict[str, str]] = None,
    labels: Optional[Dict[str, str]] = None,
    figsize: Tuple[int, int] = (12, 8),
) -> Tuple[Figure, Axes]:
    """Plot median energy spectra with 16th-84th percentile bands."""
    colors = colors or LOSS_COLORS_DEFAULT
    labels = labels or LOSS_LABELS_DEFAULT

    fig, ax = plt.subplots(figsize=figsize)

    if isinstance(targets, np.ndarray):
        targets_tensor = torch.from_numpy(targets).float()
    else:
        targets_tensor = targets.float()

    if targets_tensor.dim() == 2:
        targets_tensor = targets_tensor.unsqueeze(1)

    # Ground truth spectrum.
    freq_norm, gt_median, gt_p16, gt_p84 = compute_unbinned_spectrum(targets_tensor)
    freq_hz = normalized_freq_to_hz(freq_norm)

    ax.semilogy(freq_hz, gt_median, 'k-', linewidth=2.5, label='Ground Truth', zorder=10)
    ax.fill_between(freq_hz, gt_p16, gt_p84, color='black', alpha=0.15, zorder=9)

    for variant_name, pred in predictions.items():
        if isinstance(pred, np.ndarray):
            pred_tensor = torch.from_numpy(pred).float()
        else:
            pred_tensor = pred.float()

        # Ensure shape [N, C, T]
        if pred_tensor.dim() == 2:
            pred_tensor = pred_tensor.unsqueeze(1)

        _, pred_median, pred_p16, pred_p84 = compute_unbinned_spectrum(pred_tensor)

        color = colors.get(variant_name, 'gray')
        label = labels.get(variant_name, variant_name)

        ax.semilogy(freq_hz, pred_median, color=color, linewidth=2, label=label, alpha=0.9)
        ax.fill_between(freq_hz, pred_p16, pred_p84, color=color, alpha=0.15)

    ax.set_xlabel('Frequency (Hz)', fontsize=14)
    ax.set_ylabel('Power Spectrum E(f)', fontsize=14)
    plot_title = title or 'Energy Spectrum Comparison'
    ax.set_title(
        f'{plot_title}\nLines: Median, Bands: 16th-84th Percentile',
        fontsize=16, fontweight='bold'
    )
    ax.legend(loc='upper right', fontsize=11, framealpha=0.9)
    ax.set_xlim((0, NYQUIST_FREQ_HZ))
    ax.grid(True, alpha=0.3, which='both')
    ax.tick_params(labelsize=12)

    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

    return fig, ax


def plot_training_curves(
    history_paths: Union[str, Dict[str, str]],
    output_path: Optional[str] = None,
    title: Optional[str] = None,
    colors: Optional[Dict[str, str]] = None,
    labels: Optional[Dict[str, str]] = None,
    figsize: Tuple[int, int] = (14, 5),
) -> Tuple[Figure, np.ndarray]:
    """Plot train loss and test L2 curves from training_history.npz files."""
    colors = colors or LOSS_COLORS_DEFAULT
    labels = labels or LOSS_LABELS_DEFAULT

    if isinstance(history_paths, str):
        history_paths = {'model': history_paths}

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    for variant_name, path in history_paths.items():
        if not Path(path).exists():
            print(f"Warning: {path} not found, skipping")
            continue

        history = np.load(path)
        epochs = history['epochs']
        train_loss = history['train_loss']

        color = colors.get(variant_name, 'gray')
        label = labels.get(variant_name, variant_name)

        # Panel 1: Training Loss
        axes[0].plot(epochs, train_loss, color=color, linewidth=2, label=label)

        # Panel 2: Test L2 Error
        if 'test_l2' in history:
            test_epochs = history.get('test_epochs', epochs)
            test_l2 = history['test_l2']
            axes[1].plot(test_epochs, test_l2, color=color, linewidth=2, label=label)

    # Format Panel 1
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training Loss', fontsize=13, fontweight='bold')
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_yscale('log')

    # Format Panel 2
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Relative L2 Error', fontsize=12)
    axes[1].set_title('Test L2 Error', fontsize=13, fontweight='bold')
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)

    plot_title = title or 'Training Curves'
    fig.suptitle(plot_title, fontsize=15, fontweight='bold')
    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

    return fig, axes


def plot_combined_derivative_grid(
    all_predictions: Dict[str, Dict[str, np.ndarray]],
    targets: np.ndarray,
    output_path: str,
    mode: str = 'best',
    title: Optional[str] = None,
    arch_order: Optional[list] = None,
    loss_order: Optional[list] = None,
    arch_labels: Optional[Dict[str, str]] = None,
    loss_labels: Optional[Dict[str, str]] = None,
    dt: float = 0.02,
    figsize: Tuple[int, int] = (28, 44),
    use_interpolation: bool = True,
) -> Optional[Tuple[Figure, np.ndarray]]:
    """Plot time response, spectrum, du/dt, and d2u/dt2 for each model row."""
    from .spectral_analysis import (compute_temporal_gradient,
                                    compute_temporal_gradient_interpolated,
                                    compute_temporal_laplacian,
                                    compute_temporal_laplacian_interpolated)

    if use_interpolation:
        grad_fn = compute_temporal_gradient_interpolated
        lap_fn = compute_temporal_laplacian_interpolated
    else:
        grad_fn = compute_temporal_gradient
        lap_fn = compute_temporal_laplacian

    # Set defaults
    arch_order = arch_order or ['deeponet', 'deepokan', 'fno', 'cno']
    loss_order = loss_order or ['baseline', 'bsp']
    arch_labels = arch_labels or ARCH_LABELS
    loss_labels = loss_labels or LOSS_LABELS_DEFAULT

    model_rows = []
    for arch in arch_order:
        if arch not in all_predictions:
            continue
        for loss in loss_order:
            if loss in all_predictions[arch]:
                model_rows.append((arch, loss, all_predictions[arch][loss]))

    if len(model_rows) == 0:
        print("Warning: No model predictions available for combined grid plot")
        return None

    n_rows = len(model_rows)
    n_cols = 4

    # Adjust figure height based on actual number of rows
    fig_height = figsize[1] * n_rows / 8  # Scale proportionally
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(figsize[0], fig_height))

    if n_rows == 1:
        axes = axes[np.newaxis, :]

    n_timesteps = targets.shape[-1]
    t = np.arange(n_timesteps) * dt

    col_titles = ['Time Response u(t)', 'Energy Spectrum E(k)',
                  'Gradient du/dt', 'Laplacian d²u/dt²']

    for row_idx, (arch, loss, predictions) in enumerate(model_rows):
        sample_indices, sample_labels = select_visualization_samples(predictions, targets, n_samples=2)

        if mode == 'best':
            sample_idx = sample_indices[0]
            sample_label = sample_labels[0]
        else:
            sample_idx = sample_indices[1] if len(sample_indices) > 1 else sample_indices[0]
            sample_label = sample_labels[1] if len(sample_labels) > 1 else sample_labels[0]

        pred_1d = _extract_sample(predictions, sample_idx)
        target_1d = _extract_sample(targets, sample_idx)

        # Compute derivatives
        pred_grad = grad_fn(pred_1d, dt)
        pred_lap = lap_fn(pred_1d, dt)
        target_grad = grad_fn(target_1d, dt)
        target_lap = lap_fn(target_1d, dt)

        freqs_pred, energy_pred = compute_fft_spectrum(pred_1d, dt)
        freqs_gt, energy_gt = compute_fft_spectrum(target_1d, dt)

        arch_label = arch_labels.get(arch, arch)
        loss_label = loss_labels.get(loss, loss)
        row_label = f"{arch_label}\n({loss_label})\n{sample_label}"

        color = ARCH_COLORS.get(arch, 'gray')

        # Column 0: Time Response
        ax = axes[row_idx, 0]
        ax.plot(t, target_1d, 'k--', linewidth=1.5, alpha=0.7, label='GT')
        ax.plot(t, pred_1d, color=color, linewidth=1.2, alpha=0.9, label='Pred')
        ax.set_ylabel(row_label, fontsize=18, fontweight='bold', rotation=90,
                      ha='center', va='center', labelpad=15)
        ax.set_xlabel('Time (s)', fontsize=16)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=14)
        if row_idx == 0:
            ax.legend(loc='upper right', fontsize=14)

        # Column 1: Energy Spectrum 
        ax = axes[row_idx, 1]
        ax.semilogy(freqs_gt, energy_gt, 'k--', linewidth=1.5, alpha=0.7)
        ax.semilogy(freqs_pred, energy_pred, color=color, linewidth=1.2, alpha=0.9)
        ax.set_xlabel('Frequency (Hz)', fontsize=16)
        ax.set_xlim((0, NYQUIST_FREQ_HZ))
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=14)

        # Column 2: Gradient du/dt
        ax = axes[row_idx, 2]
        ax.plot(t, target_grad, 'k--', linewidth=1.5, alpha=0.7)
        ax.plot(t, pred_grad, color=color, linewidth=1.2, alpha=0.9)
        ax.set_xlabel('Time (s)', fontsize=16)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=14)

        # Column 3: Laplacian d²u/dt²
        ax = axes[row_idx, 3]
        ax.plot(t, target_lap, 'k--', linewidth=1.5, alpha=0.7)
        ax.plot(t, pred_lap, color=color, linewidth=1.2, alpha=0.9)
        ax.set_xlabel('Time (s)', fontsize=16)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=14)

    for col_idx, col_title in enumerate(col_titles):
        axes[0, col_idx].set_title(col_title, fontsize=20, fontweight='bold', pad=15)

    # Overall title
    mode_str = {'best': 'Best', 'worst': 'Worst', 'median': 'Median'}.get(mode, mode.title())
    plot_title = title or f'Combined Derivative Grid - {mode_str} Sample per Model'
    fig.suptitle(plot_title, fontsize=24, fontweight='bold', y=0.995)

    plt.tight_layout(rect=(0.05, 0, 1, 0.99))

    # Save
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

    return fig, axes


def _plot_arch_comparison_cell(
    ax_time: Axes,
    ax_spec: Axes,
    pred_1d: np.ndarray,
    target_1d: np.ndarray,
    t: np.ndarray,
    color: str,
    dt: float = 0.02,
    show_legend: bool = False,
) -> None:
    """Plot time-domain and spectrum cells for one architecture/loss combination."""
    freqs_pred, energy_pred = compute_fft_spectrum(pred_1d, dt)
    freqs_gt, energy_gt = compute_fft_spectrum(target_1d, dt)

    # Time Domain Response
    ax_time.plot(t, target_1d, 'k--', linewidth=2.0, alpha=0.7, label='Ground Truth')
    ax_time.plot(t, pred_1d, color=color, linewidth=1.8, alpha=0.9, label='Prediction')
    ax_time.set_xlabel('Time (s)', fontsize=18)
    ax_time.set_ylabel('Displacement (m)', fontsize=14)
    ax_time.grid(True, alpha=0.3)
    ax_time.tick_params(labelsize=14, width=1.5, length=6)

    if show_legend:
        ax_time.legend(loc='upper right', fontsize=14)

    # Energy Spectrum 
    ax_spec.semilogy(freqs_gt, energy_gt, 'k--', linewidth=2.0, alpha=0.7)
    ax_spec.semilogy(freqs_pred, energy_pred, color=color, linewidth=1.8, alpha=0.9)
    ax_spec.set_xlabel('Frequency (Hz)', fontsize=18)
    ax_spec.set_ylabel('E(f)', fontsize=14)
    ax_spec.set_xlim(0, NYQUIST_FREQ_HZ)
    ax_spec.grid(True, alpha=0.3)
    ax_spec.tick_params(labelsize=14, width=1.5, length=6)


def plot_arch_loss_comparison_grid(
    all_predictions: Dict[str, Dict[str, np.ndarray]],
    targets: np.ndarray,
    output_path: str,
    mode: str = 'best',
    title: Optional[str] = None,
    arch_order: Optional[list] = None,
    arch_labels: Optional[Dict[str, str]] = None,
    dt: float = 0.02,
    figsize: Tuple[int, int] = (24, 20),
) -> Optional[Tuple[Figure, np.ndarray]]:
    """Plot architecture rows against Baseline and BSP columns."""
    arch_order = arch_order or ['deeponet', 'deepokan', 'fno', 'cno']
    arch_labels = arch_labels or ARCH_LABELS
    loss_order = ['baseline', 'bsp']

    available_archs = [a for a in arch_order if a in all_predictions]

    if len(available_archs) == 0:
        print("Warning: No architectures available for comparison grid")
        return None

    n_rows = len(available_archs)
    n_cols = 4  

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)

    if n_rows == 1:
        axes = axes[np.newaxis, :]

    # Time axis
    n_timesteps = targets.shape[-1]
    t = np.arange(n_timesteps) * dt

    for row_idx, arch in enumerate(available_archs):
        arch_label = arch_labels.get(arch, arch)
        color = ARCH_COLORS.get(arch, 'gray')

        for loss_idx, loss in enumerate(loss_order):
            col_offset = loss_idx * 2  # 0 for baseline, 2 for bsp

            if loss not in all_predictions[arch]:
                axes[row_idx, col_offset].text(0.5, 0.5, 'N/A', ha='center', va='center',
                                                transform=axes[row_idx, col_offset].transAxes, fontsize=20)
                axes[row_idx, col_offset + 1].text(0.5, 0.5, 'N/A', ha='center', va='center',
                                                    transform=axes[row_idx, col_offset + 1].transAxes, fontsize=20)
                continue

            predictions = all_predictions[arch][loss]

            sample_indices, sample_labels = select_visualization_samples(predictions, targets, n_samples=3)

            if mode == 'best':
                sample_idx = sample_indices[0]
            elif mode == 'median':
                sample_idx = sample_indices[2] if len(sample_indices) > 2 else sample_indices[0]
            else:
                sample_idx = sample_indices[1] if len(sample_indices) > 1 else sample_indices[0]

            pred_1d = _extract_sample(predictions, sample_idx)
            target_1d = _extract_sample(targets, sample_idx)

            # Plot time-domain and spectrum cells
            ax_time = axes[row_idx, col_offset]
            ax_spec = axes[row_idx, col_offset + 1]
            show_legend = (row_idx == 0 and loss_idx == 0)
            _plot_arch_comparison_cell(
                ax_time, ax_spec, pred_1d, target_1d, t, color, dt,
                show_legend=show_legend,
            )

        axes[row_idx, 0].text(
            -0.35, 0.5, arch_label, transform=axes[row_idx, 0].transAxes,
            fontsize=18, fontweight='bold', rotation=90,
            ha='center', va='center')

    col_subtitles = ['Time Response', 'Energy Spectrum', 'Time Response', 'Energy Spectrum']
    for col_idx, subtitle in enumerate(col_subtitles):
        axes[0, col_idx].set_title(subtitle, fontsize=18, pad=15)

    plt.tight_layout(rect=(0.08, 0, 1, 0.94))

    ax0_pos = axes[0, 0].get_position()
    ax1_pos = axes[0, 1].get_position()
    ax2_pos = axes[0, 2].get_position()
    ax3_pos = axes[0, 3].get_position()

    # Baseline centered between columns 0-1
    baseline_x = (ax0_pos.x0 + ax1_pos.x1) / 2
    # BSP centered between columns 2-3
    bsp_x = (ax2_pos.x0 + ax3_pos.x1) / 2
    # Y position just above the column subtitles (closer to plots)
    header_y = ax0_pos.y1 + 0.025

    fig.text(baseline_x, header_y, 'Baseline', fontsize=22, fontweight='bold',
             ha='center', va='bottom')
    fig.text(bsp_x, header_y, 'BSP', fontsize=22, fontweight='bold',
             ha='center', va='bottom')

    mode_str = {'best': 'Best', 'worst': 'Worst', 'median': 'Median'}.get(mode, mode.title())
    plot_title = title or f'Architecture Comparison - {mode_str} nRMSE Sample per Model'
    fig.suptitle(plot_title, fontsize=24, fontweight='bold', y=0.99, x=0.54, ha='center')

    # Save
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

    return fig, axes


def plot_fixed_sample_comparison_grid(
    rows: List[Dict],
    targets: np.ndarray,
    sample_idx: int,
    output_path: str,
    group_a_label: str = 'Baseline',
    group_b_label: str = 'BSP',
    title: Optional[str] = None,
    dt: float = 0.02,
    figsize: Optional[Tuple[int, int]] = None,
) -> Optional[Tuple[Figure, np.ndarray]]:
    """Plot time/spectrum comparisons for a fixed sample across all rows."""
    if not rows:
        print("Warning: No rows provided for comparison grid")
        return None

    n_rows = len(rows)
    n_cols = 4  

    if figsize is None:
        figsize = (24, 5 * n_rows + 2)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)

    if n_rows == 1:
        axes = axes[np.newaxis, :]

    n_timesteps = targets.shape[-1]
    t = np.arange(n_timesteps) * dt

    target_1d = _extract_sample(targets, sample_idx)

    for row_idx, row in enumerate(rows):
        label = row['label']
        color = row['color']

        for group_idx, variant_key in enumerate(['variant_a', 'variant_b']):
            col_offset = group_idx * 2  # 0 for group_a, 2 for group_b
            preds = row.get(variant_key)

            if preds is None:
                axes[row_idx, col_offset].text(
                    0.5, 0.5, 'N/A', ha='center', va='center',
                    transform=axes[row_idx, col_offset].transAxes, fontsize=20)
                axes[row_idx, col_offset + 1].text(
                    0.5, 0.5, 'N/A', ha='center', va='center',
                    transform=axes[row_idx, col_offset + 1].transAxes, fontsize=20)
                continue

            pred_1d = _extract_sample(preds, sample_idx)

            ax_time = axes[row_idx, col_offset]
            ax_spec = axes[row_idx, col_offset + 1]
            show_legend = (row_idx == 0 and group_idx == 0)

            _plot_arch_comparison_cell(
                ax_time, ax_spec, pred_1d, target_1d, t, color, dt,
                show_legend=show_legend,
            )

        axes[row_idx, 0].text(
            -0.35, 0.5, label, transform=axes[row_idx, 0].transAxes,
            fontsize=18, fontweight='bold', rotation=90,
            ha='center', va='center')

    col_subtitles = ['Time Response', 'Energy Spectrum',
                     'Time Response', 'Energy Spectrum']
    for col_idx, subtitle in enumerate(col_subtitles):
        axes[0, col_idx].set_title(subtitle, fontsize=18, pad=15)

    # Layout then add group headers — leave more top margin for headers
    plt.tight_layout(rect=(0.08, 0, 1, 0.91))

    ax0_pos = axes[0, 0].get_position()
    ax1_pos = axes[0, 1].get_position()
    ax2_pos = axes[0, 2].get_position()
    ax3_pos = axes[0, 3].get_position()

    group_a_x = (ax0_pos.x0 + ax1_pos.x1) / 2
    group_b_x = (ax2_pos.x0 + ax3_pos.x1) / 2
    header_y = ax0_pos.y1 + 0.055

    fig.text(group_a_x, header_y, group_a_label, fontsize=22,
             fontweight='bold', ha='center', va='bottom')
    fig.text(group_b_x, header_y, group_b_label, fontsize=22,
             fontweight='bold', ha='center', va='bottom')

    if title:
        fig.suptitle(title, fontsize=24, fontweight='bold', y=0.99,
                     x=0.54, ha='center')

    # Save
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

    return fig, axes


def plot_compact_impact_grid(
    rows: List[Dict],
    targets: List[np.ndarray],
    output_path: str,
    dt: float = 0.02,
    figsize: Tuple[int, int] = (10, 10),
) -> Optional[Tuple[Figure, np.ndarray]]:
    """Generate compact 2-column x N-row grid: Time Response | Energy Spectrum.

    Each row overlays multiple predictions (colored) on top of a shared
    ground truth (black dashed).  Designed as a compact summary figure
    for the paper's main text.

    Args:
        rows: List of row dicts, each with:
            - 'label': Row label (displayed on left y-axis, rotated 90 deg)
            - 'variants': List of (name, color, linestyle, pred_1d) tuples
        targets: List of target arrays [T], one per row
        output_path: Path to save the figure
        dt: Time step in seconds (default: 0.02)
        figsize: Figure size (width, height)

    Returns:
        (fig, axes) tuple, or None if no valid rows
    """
    if not rows:
        print("Warning: No rows provided for compact impact grid")
        return None

    n_rows = len(rows)
    fig, axes = plt.subplots(n_rows, 2, figsize=figsize)

    if n_rows == 1:
        axes = axes[np.newaxis, :]

    axes[0, 0].set_title('Time Response', fontsize=14, fontweight='bold', pad=25)
    axes[0, 1].set_title('Energy Spectrum', fontsize=14, fontweight='bold', pad=25)

    for row_idx, (row, target_1d) in enumerate(zip(rows, targets)):
        variants = row['variants']
        t = np.arange(len(target_1d)) * dt

        # Column 0: Time Response
        ax_time = axes[row_idx, 0]
        ax_time.plot(t, target_1d, 'k--', linewidth=2.0, alpha=0.7, label='Ground Truth')
        for name, color, linestyle, pred_1d in variants:
            ax_time.plot(t, pred_1d, color=color, linestyle=linestyle,
                         linewidth=1.5, alpha=0.85, label=name)
        ax_time.set_xlabel('Time (s)', fontsize=11)
        ax_time.set_ylabel('Displacement (m)', fontsize=10)
        ax_time.grid(True, alpha=0.3)
        ax_time.tick_params(labelsize=10)
        ax_time.legend(loc='upper right', fontsize=8, framealpha=0.9)

        # Column 1: Energy Spectrum 
        ax_spec = axes[row_idx, 1]
        freqs_gt, energy_gt = compute_fft_spectrum(target_1d, dt)
        ax_spec.semilogy(freqs_gt, energy_gt, 'k--', linewidth=2.0, alpha=0.7)
        for name, color, linestyle, pred_1d in variants:
            freqs, energy = compute_fft_spectrum(pred_1d, dt)
            ax_spec.semilogy(freqs, energy, color=color, linestyle=linestyle,
                             linewidth=1.5, alpha=0.85)
        ax_spec.set_xlabel('Frequency (Hz)', fontsize=11)
        ax_spec.set_ylabel('E(f)', fontsize=10)
        ax_spec.set_xlim(0, NYQUIST_FREQ_HZ)
        ax_spec.grid(True, alpha=0.3)
        ax_spec.tick_params(labelsize=10)

    for row_idx, row in enumerate(rows):
        letter = chr(ord('a') + row_idx)
        label_text = row['label'].replace('\n', ' ')
        axes[row_idx, 0].text(
            0.0, 1.01, f'({letter}) {label_text}',
            transform=axes[row_idx, 0].transAxes,
            fontsize=12, fontweight='bold', va='bottom', ha='left',
        )

    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

    return fig, axes
