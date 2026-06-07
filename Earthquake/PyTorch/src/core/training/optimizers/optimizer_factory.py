"""
Optimizer factory.

Supports:
- Adam: Standard Adam optimizer
- AdamW: Adam with decoupled weight decay
- SOAP: Shampoo with Adam in the Preconditioner
"""

from typing import TYPE_CHECKING

import torch.optim as optim

if TYPE_CHECKING:
    from configs.training_config import TrainingConfig

from .soap import SOAP


def create_optimizer(
    optimizer_type: str, model_parameters, config: "TrainingConfig"
) -> optim.Optimizer:
    """
    Factory function.

    Args:
        optimizer_type: Type of optimizer ('adam', 'adamw', 'soap')
        model_parameters: Model parameters to optimize (from model.parameters())
        config: Training configuration with optimizer hyperparameters

    Returns:
        Initialized optimizer instance

    Raises:
        ValueError: If optimizer_type is not supported

    Examples:
        >>> from configs.training_config import TrainingConfig
        >>> config = TrainingConfig(optimizer_type='adam', learning_rate=1e-3)
        >>> optimizer = create_optimizer('adam', model.parameters(), config)
    """
    optimizer_type = optimizer_type.lower()

    if optimizer_type == "adam":
        return optim.Adam(
            model_parameters, lr=config.learning_rate, weight_decay=config.weight_decay
        )

    elif optimizer_type == "adamw":
        return optim.AdamW(
            model_parameters, lr=config.learning_rate, weight_decay=config.weight_decay
        )

    elif optimizer_type == "soap":
        return SOAP(
            model_parameters,
            lr=config.learning_rate,
            betas=config.soap_betas,
            shampoo_beta=config.soap_shampoo_beta,
            eps=config.soap_eps,
            weight_decay=config.weight_decay,
            precondition_frequency=config.soap_precondition_frequency,
            max_precond_dim=config.soap_max_precond_dim,
            merge_dims=config.soap_merge_dims,
            precondition_1d=config.soap_precondition_1d,
            normalize_grads=config.soap_normalize_grads,
        )

    else:
        raise ValueError(
            f"Unsupported optimizer type: '{optimizer_type}'. "
            "Supported types: 'adam', 'adamw', 'soap'"
        )
