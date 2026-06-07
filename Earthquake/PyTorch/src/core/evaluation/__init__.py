"""Loss functions and evaluation metrics."""
from .loss_factory import create_loss
from .metrics import (
    EvaluationResults,
    compute_acceleration_nrmse,
    compute_barron_norm_error,
    compute_barron_spectrum,
    compute_derivative_metrics,
    compute_energy_spectrum,
    compute_field_nrmse,
    compute_log_spectral_nrmse,
    compute_log_spectral_quintile_errors,
    compute_nrmse,
    compute_velocity_nrmse,
    evaluate_model,
)
from .utils import get_model_predictions

__all__ = [
    # Loss functions
    "create_loss",
    # Metrics (global normalization)
    "EvaluationResults",
    "compute_nrmse",
    "compute_field_nrmse",
    "compute_log_spectral_nrmse",
    "compute_log_spectral_quintile_errors",
    "compute_energy_spectrum",
    "compute_barron_spectrum",
    "compute_barron_norm_error",
    "compute_velocity_nrmse",
    "compute_acceleration_nrmse",
    "compute_derivative_metrics",
    "evaluate_model",
    # Utilities
    "get_model_predictions",
]
