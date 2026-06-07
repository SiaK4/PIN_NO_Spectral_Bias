"""
Constants for evaluation and loss functions.
"""

import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np

# Global nRMSE Statistics (dataset-wide min/max for normalization)
# Path to precomputed global stats (relative to project root)
GLOBAL_NRMSE_STATS_PATH = (
    Path(__file__).parent.parent.parent.parent / "configs" / "global_nrmse_stats.json"
)


def load_global_nrmse_stats() -> Optional[Dict[str, Dict[str, float]]]:
    """
    Load global nRMSE statistics from precomputed JSON file.

    Returns:
        Dictionary with global stats for each metric type, or None if file not found.
        Format: {'field': {'global_min': X, 'global_max': Y}, ...}
    """
    if not GLOBAL_NRMSE_STATS_PATH.exists():
        return None

    with open(GLOBAL_NRMSE_STATS_PATH, "r") as f:
        return json.load(f)


# Load global stats at module import (cached)
GLOBAL_NRMSE_STATS: Optional[Dict[str, Dict[str, float]]] = load_global_nrmse_stats()


# Numerical Stability
EPSILON = 1e-8

# Epsilon for spectral computations: log transforms and sqrt in Barron norm.
# Used with float64 precision.
EPSILON_SPECTRAL = 1e-30


# BSP weight in the combined loss: L = L_field + w_bsp * L_spectral
W_BSP = 1.0


# Spectral Analysis

# Physical domain length for CDON spectral analysis
LX_CDON = 2 * np.pi

# CDON dataset signal length (all temporal signals are 4000 timesteps)
SIGNAL_LENGTH_CDON = 4000

# CNO internal processing resolution (requires pre-interpolation)
# CNO processes at 4096 internally, then post-processes back to 4000 for evaluation
CNO_INTERNAL_SIZE = 4096


# Batch Sizes

# Per-timestep batch size for baseline (field-only) DeepONet/DeepOKAN training
# Uses flat per-timestep DataLoader - samples shuffled across all earthquakes
BATCH_SIZE_PER_TIMESTEP_BASELINE = 50000


# Sequence batch size (for full sequence training and BSP losses)
# Used by all architectures: FNO/CNO for field+BSP, DeepONet/DeepOKAN for BSP
# 8 sequences × 4001 timesteps = 32,008 per-timestep samples for DeepONet
BATCH_SIZE_SEQUENCE = 8


# Model Architecture Defaults

# SIREN w0 parameter (initial frequency scale)
# w0=30 is the standard value from the SIREN paper (Sitzmann et al. 2020)
# Higher w0 → higher frequency signals, may cause instability
# Lower w0 → smoother outputs, more stable but less expressive
SIREN_W0_DEFAULT = 30.0

# Default dropout rate for neural operator models
DROPOUT_DEFAULT = 0.0

# Default activation for DeepONet branch network
# 'tanh' is recommended for stable input encoding
BRANCH_ACTIVATION_DEFAULT = "tanh"

# Default activation for DeepONet trunk network
# 'siren' is recommended for smooth coordinate interpolation
TRUNK_ACTIVATION_DEFAULT = "siren"
