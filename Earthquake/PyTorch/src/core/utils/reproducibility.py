"""Reproducibility utilities for deterministic training."""

import os
import random
from typing import Optional

import numpy as np
import torch


def set_global_seed(
    seed: Optional[int] = 42, deterministic: bool = True, verbose: bool = True
) -> None:
    """Set random seeds for reproducibility across all libraries.

    This function sets seeds for Python's random module, NumPy, and PyTorch
    (including CUDA if available). It also configures cuDNN for deterministic
    behavior when requested.

    Args:
        seed: Random seed for reproducibility. If None, uses non-deterministic
            behavior (no seeds set, benchmark mode enabled).
        deterministic: If True, use deterministic algorithms. This may be slower
            but ensures reproducibility. Only applies when seed is not None.
        verbose: If True, print seed configuration info.

    Example:
        >>> set_global_seed(1)  # Standard reproducible setup
        >>> set_global_seed(1, deterministic=False)  # Reproducible but allow non-deterministic ops
        >>> set_global_seed(None)  # Non-deterministic mode for best performance
    """
    if seed is None:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        if verbose:
            print("Running in non-deterministic mode (no seed set, benchmark enabled)")
        return

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU setups

    os.environ["PYTHONHASHSEED"] = str(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

    if verbose:
        print(f"Set global seed: {seed}, deterministic: {deterministic}")


def worker_init_fn(worker_id: int) -> None:
    """Seed a DataLoader worker's numpy/random from the main-process seed.

    Pass as DataLoader(worker_init_fn=...) so each worker gets a unique but
    reproducible seed (main seed + worker_id).
    """
    worker_seed = torch.initial_seed() % 2**32

    np.random.seed(worker_seed + worker_id)
    random.seed(worker_seed + worker_id)
