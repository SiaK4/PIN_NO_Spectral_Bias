"""Visualization utilities for spectral analysis and evaluation plots."""

from .spectral_analysis import (
    compute_unbinned_spectrum,
    normalized_freq_to_hz,
    compute_temporal_gradient,
    compute_temporal_laplacian,
)

from .plotting_utils import (
    # Sample selection utilities
    select_visualization_samples,
    # Core plotting functions
    compute_fft_spectrum,
    plot_4panel_comparison,
    plot_energy_spectrum_comparison,
    plot_training_curves,
    plot_combined_derivative_grid,
    plot_arch_loss_comparison_grid,
    plot_fixed_sample_comparison_grid,
    plot_compact_impact_grid,
    # Color schemes
    LOSS_COLORS_DEFAULT,
    LOSS_LABELS_DEFAULT,
    ACTIVATION_COLORS,
    ACTIVATION_LABELS,
    OPTIMIZER_COLORS,
    OPTIMIZER_LABELS,
    ARCH_COLORS,
    ARCH_LABELS,
)

__all__ = [
    # Spectral analysis
    'compute_unbinned_spectrum',
    'normalized_freq_to_hz',
    'compute_temporal_gradient',
    'compute_temporal_laplacian',
    # Sample selection
    'select_visualization_samples',
    # Plotting utilities
    'compute_fft_spectrum',
    'plot_4panel_comparison',
    'plot_energy_spectrum_comparison',
    'plot_training_curves',
    'plot_combined_derivative_grid',
    'plot_arch_loss_comparison_grid',
    'plot_fixed_sample_comparison_grid',
    'plot_compact_impact_grid',
    # Color/label constants
    'LOSS_COLORS_DEFAULT',
    'LOSS_LABELS_DEFAULT',
    'ACTIVATION_COLORS',
    'ACTIVATION_LABELS',
    'OPTIMIZER_COLORS',
    'OPTIMIZER_LABELS',
    'ARCH_COLORS',
    'ARCH_LABELS',
]
