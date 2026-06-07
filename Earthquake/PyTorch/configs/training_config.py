"""
Configuration dataclass for training neural operator models.

Provides structured configuration for training hyperparameters,
scheduler settings, and checkpoint management.
"""

from dataclasses import dataclass, field
from typing import List, Optional

from configs.base_config import BaseConfig

# Standardized hyperparameters

LEARNING_RATE_SOAP = 1e-3
LEARNING_RATE_ADAM = 1e-3


@dataclass
class TrainingConfig(BaseConfig):
    """
    Configuration for training neural operator models.

    Groups: Optimization (SOAP/Adam), Scheduler (cosine/none),
    Evaluation, Checkpointing, Device, and Logging.

    See inline comments for individual parameter documentation.
    """

    # Optimization
    learning_rate: float = 1e-3
    num_epochs: int = 100
    batch_size: int = 128
    weight_decay: float = 1e-4
    optimizer_type: str = "soap"
    max_grad_norm: float = 0.0

    soap_betas: tuple = (0.95, 0.95)
    soap_shampoo_beta: float = 0.95
    soap_eps: float = 1e-8
    soap_precondition_frequency: int = 10
    soap_max_precond_dim: int = 10000
    soap_merge_dims: bool = False
    soap_precondition_1d: bool = False
    soap_normalize_grads: bool = False

    # Scheduler
    scheduler_type: str = "cosine"  # 'cosine' or 'none'
    cosine_t_max: Optional[int] = None  # Will be set to num_epochs * steps_per_epoch
    cosine_eta_min: float = 0.0

    eval_metrics: List[str] = field(
        default_factory=lambda: ["relative_l2", "log_spectral_error"]
    )
    eval_frequency: int = 1

    # Checkpointing
    checkpoint_dir: str = "checkpoints/"
    save_best: bool = True
    save_latest: bool = True
    save_frequency: int = 10
    save_milestone_epochs: List[int] = field(default_factory=lambda: [100])
    save_milestone_frequency: int = 5000

    # Time limits
    max_training_time: Optional[float] = (
        None  # Max wall-clock time in seconds (None = no limit)
    )

    # Device
    device: str = "cuda"
    num_workers: int = 4

    # Logging
    log_frequency: int = 10
    verbose: bool = True

    # Reproducibility
    seed: Optional[int] = 42
    deterministic: bool = True

    def __repr__(self) -> str:
        """String representation showing key parameters."""
        return (
            f"TrainingConfig(\n"
            f"  Optimization:\n"
            f"    optimizer_type='{self.optimizer_type}',\n"
            f"    learning_rate={self.learning_rate},\n"
            f"    num_epochs={self.num_epochs},\n"
            f"    batch_size={self.batch_size},\n"
            f"    weight_decay={self.weight_decay}\n"
            f"  Scheduler:\n"
            f"    scheduler_type='{self.scheduler_type}',\n"
            f"    cosine_eta_min={self.cosine_eta_min}\n"
            f"  Evaluation:\n"
            f"    eval_metrics={self.eval_metrics}\n"
            f"  Checkpointing:\n"
            f"    checkpoint_dir='{self.checkpoint_dir}',\n"
            f"    save_best={self.save_best},\n"
            f"    save_latest={self.save_latest}\n"
            f"  Device: {self.device}\n"
            f")"
        )
