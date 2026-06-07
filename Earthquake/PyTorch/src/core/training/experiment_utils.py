#!/usr/bin/env python3
"""Shared helpers for model setup, data loading, training, and evaluation."""

import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader


def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent.parent.parent


# Add project root to path for imports
project_root = get_project_root()
sys.path.insert(0, str(project_root))

from configs.model_configs import MODEL_CONFIGS
from configs.training_config import (
    LEARNING_RATE_ADAM,
    LEARNING_RATE_SOAP,
    TrainingConfig,
)
from src.core.data_processing.cdon_dataset import CDONDataset
from src.core.evaluation.constants import (
    BATCH_SIZE_PER_TIMESTEP_BASELINE,
    BATCH_SIZE_SEQUENCE,
    CNO_INTERNAL_SIZE,
    SIGNAL_LENGTH_CDON,
)
from src.core.evaluation.metrics import evaluate_model
from src.core.evaluation.utils import get_model_predictions, postprocess_cno_predictions
from src.core.models.model_factory import create_model
from src.core.training.simple_trainer import SimpleTrainer
from src.core.utils.reproducibility import worker_init_fn


def get_run_name(
    arch: str,
    optimizer: str,
    loss: str,
    activation: Optional[str] = None,
    causal_mode: Optional[str] = None,
    suffix: Optional[str] = None,
) -> str:
    """Generate the checkpoint/results run name for one experiment variant."""
    parts = [arch]
    if arch == "deeponet" and activation and activation != "siren":
        parts.append(activation)
    if causal_mode is not None:
        parts.append(causal_mode)
    parts.extend([optimizer, loss])
    if suffix is not None:
        parts.append(suffix)
    return "_".join(parts)


def _model_config_for_variant(
    arch: str,
    activation: Optional[str] = None,
) -> Dict[str, Any]:
    """Return a model config with the requested DeepONet activation variant."""
    config = MODEL_CONFIGS[arch].copy()
    if arch == "deeponet" and activation:
        if activation == "tanh":
            config["branch_activation"] = "tanh"
            config["trunk_activation"] = "tanh"
        elif activation == "siren":
            config["branch_activation"] = "tanh"
            config["trunk_activation"] = "siren"
    return config


def _load_model_state_dict(checkpoint_path: Path, device: str) -> Dict[str, Any]:
    """Load the model state dictionary from a checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    state_dict = checkpoint["model_state_dict"]
    state_dict.pop("_metadata", None)
    return state_dict


def load_trained_model(
    arch: str,
    optimizer: str,
    loss: str,
    checkpoint_dir: Path,
    device: str = "cuda",
    activation: Optional[str] = None,
    causal_mode: Optional[str] = None,
) -> Optional[torch.nn.Module]:
    """Load a trained model by experiment variant."""
    run_name = get_run_name(arch, optimizer, loss, activation, causal_mode)
    checkpoint_path = checkpoint_dir / run_name / "best_model.pt"
    if not checkpoint_path.exists():
        print(f"  Checkpoint not found: {checkpoint_path}")
        return None

    model = create_model(arch, config=_model_config_for_variant(arch, activation))
    model.load_state_dict(_load_model_state_dict(checkpoint_path, device))
    model.to(device)
    model.eval()
    return model


# cudnn.benchmark is controlled by set_global_seed() in SimpleTrainer.
_DATALOADER_PERF_KWARGS: Dict[str, Any] = {
    "pin_memory": True,
    "persistent_workers": True,
    "prefetch_factor": 4,
    "worker_init_fn": worker_init_fn,
}


def prepare_model(
    model: torch.nn.Module,
    device: str = "cuda",
) -> torch.nn.Module:
    """
    Prepare model for training.

    Applies device transfer (model.to(device)).

    torch.compile is intentionally not used because it adds '_orig_mod.' to
    state_dict keys, causing strict checkpoint loading failures.

    Args:
        model: PyTorch model instance
        device: Target device

    Returns:
        Model ready for training
    """
    model = model.to(device)
    return model


def create_dataloaders(
    arch: str,
    batch_size: int = 16,
    num_workers: int = 4,
    data_dir: Optional[Path] = None,
    is_baseline: bool = False,
    use_causal: bool = True,
) -> Dict[str, Any]:
    """Create architecture-specific train/test dataloaders and metadata."""
    if data_dir is None:
        data_dir = project_root / "CDONData"

    is_deeponet_arch = (arch in ["deeponet", "deepokan"]) and use_causal

    perf_kwargs = _DATALOADER_PERF_KWARGS if num_workers > 0 else {"pin_memory": True}

    if is_deeponet_arch:
        per_ts_train = CDONDataset(
            data_dir=str(data_dir),
            split="train",
            mode="per_timestep",
            signal_length=SIGNAL_LENGTH_CDON,
        )
        seq_test = CDONDataset(
            data_dir=str(data_dir),
            split="test",
            mode="sequence",
            signal_length=SIGNAL_LENGTH_CDON,
        )

        if is_baseline:
            return {
                "per_timestep_train": DataLoader(
                    per_ts_train,
                    batch_size=BATCH_SIZE_PER_TIMESTEP_BASELINE,
                    shuffle=True,
                    num_workers=num_workers,
                    **perf_kwargs,
                ),
                "per_timestep_val": None,
                "test": DataLoader(
                    seq_test,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=num_workers,
                    **perf_kwargs,
                ),
                "is_deeponet": True,
                "is_baseline": True,
                "per_timestep_samples": len(per_ts_train),
            }
        else:
            seq_train = CDONDataset(
                data_dir=str(data_dir),
                split="train",
                mode="sequence",
                signal_length=SIGNAL_LENGTH_CDON,
            )

            seq_batch = BATCH_SIZE_SEQUENCE

            return {
                "per_timestep_train_dataset": per_ts_train,
                "per_timestep_val_dataset": None,
                "sequence_train": DataLoader(
                    seq_train,
                    batch_size=seq_batch,
                    shuffle=True,
                    num_workers=num_workers,
                    **perf_kwargs,
                ),
                "sequence_val": None,
                "test": DataLoader(
                    seq_test,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=num_workers,
                    **perf_kwargs,
                ),
                "is_deeponet": True,
                "is_baseline": False,
                "per_timestep_samples": len(per_ts_train),
                "sequence_samples": len(seq_train),
            }
    else:
        target_length = CNO_INTERNAL_SIZE if arch == "cno" else None

        train_dataset = CDONDataset(
            data_dir=str(data_dir),
            split="train",
            mode="sequence",
            signal_length=SIGNAL_LENGTH_CDON,
            target_signal_length=target_length,
        )
        test_dataset = CDONDataset(
            data_dir=str(data_dir),
            split="test",
            mode="sequence",
            signal_length=SIGNAL_LENGTH_CDON,
            target_signal_length=target_length,
        )

        return {
            "train": DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                **perf_kwargs,
            ),
            "val": None,
            "test": DataLoader(
                test_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                **perf_kwargs,
            ),
            "is_deeponet": False,
            "sequence_samples": len(train_dataset),
        }


def create_test_dataloader(
    test_dataset, batch_size: int = 8, num_workers: int = 4
) -> DataLoader:
    """
    Create optimized test DataLoader.

    Use this in evaluation scripts instead of creating DataLoader directly.
    Automatically applies: pin_memory, persistent_workers, prefetch_factor.

    Args:
        test_dataset: Test dataset instance
        batch_size: Batch size for evaluation
        num_workers: Number of worker processes

    Returns:
        Optimized DataLoader for evaluation
    """
    return DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        **(_DATALOADER_PERF_KWARGS if num_workers > 0 else {"pin_memory": True}),
    )


def create_trainer(
    model: torch.nn.Module,
    loaders: Dict[str, Any],
    loss_config: Any,
    experiment_name: str,
    checkpoint_dir: Path,
    epochs: int = 30,
    device: str = "cuda",
    arch: str = "deeponet",
    optimizer_override: str = None,
    lr_override: float = None,
    use_causal: bool = False,
    max_grad_norm: float = 0.0,
    scheduler_type: str = "cosine",
    max_training_time: Optional[float] = None,
    seed: int = 1,
) -> SimpleTrainer:
    """Create a SimpleTrainer for baseline or BSP training."""
    is_deeponet = loaders["is_deeponet"]
    is_baseline = loaders.get("is_baseline", False)

    if optimizer_override is not None:
        optimizer_type = optimizer_override
        learning_rate = (
            LEARNING_RATE_ADAM if optimizer_type == "adam" else LEARNING_RATE_SOAP
        )
    else:
        optimizer_type = "soap"
        learning_rate = LEARNING_RATE_SOAP

    if lr_override is not None:
        learning_rate = lr_override

    # Batch size for config (used for logging, actual batch determined by loaders)
    if arch == "deeponet":
        batch_size = 16
    else:
        batch_size = 8

    train_config = TrainingConfig(
        learning_rate=learning_rate,
        num_epochs=epochs,
        batch_size=batch_size,
        optimizer_type=optimizer_type,
        device=device,
        checkpoint_dir=str(checkpoint_dir),
        eval_metrics=["field_nrmse", "log_spectral_nrmse"],
        max_grad_norm=max_grad_norm,
        scheduler_type=scheduler_type,
        max_training_time=max_training_time,
        seed=seed,
    )

    test_loader = loaders.get("test")

    if is_deeponet:
        if is_baseline:
            return SimpleTrainer(
                model=model,
                per_timestep_train_loader=loaders["per_timestep_train"],
                sequence_train_loader=None,
                config=train_config,
                loss_config=loss_config,
                experiment_name=experiment_name,
                test_loader=test_loader,
            )
        else:
            per_ts_train_dataset = loaders["per_timestep_train_dataset"]

            per_ts_train_loader = DataLoader(
                per_ts_train_dataset, batch_size=1, shuffle=False
            )

            return SimpleTrainer(
                model=model,
                per_timestep_train_loader=per_ts_train_loader,
                sequence_train_loader=loaders["sequence_train"],
                config=train_config,
                loss_config=loss_config,
                experiment_name=experiment_name,
                test_loader=test_loader,
            )
    else:
        return SimpleTrainer(
            model=model,
            train_loader=loaders["train"],
            config=train_config,
            loss_config=loss_config,
            experiment_name=experiment_name,
            test_loader=test_loader,
            use_causal=use_causal,
        )


def load_best_model(
    model: torch.nn.Module, checkpoint_dir: Path, run_name: str, device: str = "cuda"
) -> torch.nn.Module:
    """
    Load the best model checkpoint.

    Args:
        model: Model instance to load weights into
        checkpoint_dir: Base checkpoint directory
        run_name: Name of the experiment run
        device: Device to load model to

    Returns:
        Model with loaded weights
    """
    best_checkpoint = checkpoint_dir / run_name / "best_model.pt"
    if best_checkpoint.exists():
        model.load_state_dict(_load_model_state_dict(best_checkpoint, device))

    model.eval()
    model.to(device)
    return model


def evaluate_on_test_set(
    model: torch.nn.Module,
    test_loader: DataLoader,
    device: str = "cuda",
    use_causal: bool = False,
    return_inputs: bool = False,
    arch: Optional[str] = None,
) -> Tuple:
    """
    Evaluate on the test set and return predictions plus metrics.

    Metrics are computed in physical space. Sample index 0 is skipped, and CNO
    outputs are interpolated from 4096 to 4000 when arch='cno'.
    """
    all_preds_physical = []
    all_targets_physical = []
    all_inputs_physical = [] if return_inputs else None
    all_indices = []

    with torch.no_grad():
        for batch in test_loader:
            inputs, targets, sample_indices = batch
            inputs = inputs.to(device)
            targets = targets.to(device)

            preds = get_model_predictions(model, inputs, use_causal=use_causal)

            all_preds_physical.append(preds.cpu())
            all_targets_physical.append(targets.cpu())
            if return_inputs:
                all_inputs_physical.append(inputs.cpu())

            all_indices.append(sample_indices.cpu())

    all_preds_physical = torch.cat(all_preds_physical, dim=0)
    all_targets_physical = torch.cat(all_targets_physical, dim=0)
    all_indices = torch.cat(all_indices, dim=0)

    if return_inputs:
        all_inputs_physical = torch.cat(all_inputs_physical, dim=0)

    # CNO post-processing: interpolate from 4096 to 4000 before metric computation
    if arch == "cno":
        all_preds_physical = postprocess_cno_predictions(all_preds_physical)
        all_targets_physical = postprocess_cno_predictions(all_targets_physical)
        if return_inputs and all_inputs_physical is not None:
            all_inputs_physical = postprocess_cno_predictions(all_inputs_physical)

    # Skip sample 0 (zero signal pair) for metrics computation
    non_zero_mask = all_indices != 0

    preds_eval_physical = all_preds_physical[non_zero_mask]
    targets_eval_physical = all_targets_physical[non_zero_mask]
    eval_results_physical = evaluate_model(preds_eval_physical, targets_eval_physical)

    eval_results = {
        "physical": eval_results_physical,
    }

    if return_inputs:
        return (
            None,
            None,
            all_preds_physical,
            all_targets_physical,
            all_inputs_physical,
            eval_results,
        )
    else:
        return None, None, all_preds_physical, all_targets_physical, eval_results


def evaluate_and_get_metrics(
    model: torch.nn.Module,
    test_loader: DataLoader,
    device: str = "cuda",
    use_causal: bool = False,
    arch: Optional[str] = None,
) -> Dict[str, Any]:
    """Evaluate on the test set and return only metrics."""
    _, _, _, _, eval_results = evaluate_on_test_set(
        model, test_loader, device, use_causal, arch=arch
    )
    return eval_results


def save_results(results: Dict[str, Any], checkpoint_dir: Path, run_name: str) -> Path:
    """
    Save evaluation results to JSON.

    Args:
        results: Dictionary of results to save
        checkpoint_dir: Base checkpoint directory
        run_name: Name of the experiment run

    Returns:
        Path to saved results file
    """
    results_path = checkpoint_dir / run_name / "eval_results.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)

    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    return results_path


def save_training_history(
    trainer,
    checkpoint_dir: Path,
    run_name: str,
    use_causal: bool = False,
    extra_fields: Optional[Dict[str, Any]] = None,
    epochs_completed: Optional[int] = None,
    training_time_seconds: Optional[float] = None,
) -> Path:
    """
    Save training history in standardized format.

    Extracts training and test history from the trainer and saves to .npz file.
    Handles both baseline (field-only) and BSP (field + BSP components) training.

    Args:
        trainer: SimpleTrainer instance with train_history and test_history attributes
        checkpoint_dir: Base checkpoint directory
        run_name: Name of the experiment run
        use_causal: Whether causal training was used
        extra_fields: Additional fields to save (e.g., metric_domain)
        epochs_completed: Number of epochs completed (defaults to len(train_history))
        training_time_seconds: Total training time in seconds (defaults to 0.0)

    Returns:
        Path to saved history file
    """
    train_hist = getattr(trainer, "train_history", [])
    test_hist = getattr(trainer, "test_history", [])

    effective_epochs = (
        epochs_completed if epochs_completed is not None else len(train_hist)
    )
    effective_time = training_time_seconds if training_time_seconds is not None else 0.0

    history_data = {
        "epochs": np.arange(1, len(train_hist) + 1),
        "train_loss": np.array([h.get("loss", 0.0) for h in train_hist]),
        "use_causal": use_causal,
        "epochs_completed": effective_epochs,
        "training_time_seconds": effective_time,
    }

    if train_hist and "field_loss" in train_hist[0]:
        history_data["train_field_loss"] = np.array(
            [h.get("field_loss", 0.0) for h in train_hist]
        )
    if train_hist and "bsp_loss" in train_hist[0]:
        history_data["train_bsp_loss"] = np.array(
            [h.get("bsp_loss", 0.0) for h in train_hist]
        )

    if test_hist:
        history_data["test_epochs"] = np.array(
            [h.get("epoch", i + 1) for i, h in enumerate(test_hist)]
        )
        history_data["test_l2"] = np.array([h.get("test_l2", 0.0) for h in test_hist])

    if extra_fields:
        history_data.update(extra_fields)

    history_path = checkpoint_dir / run_name / "training_history.npz"
    history_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(history_path, **history_data)
    print(f"   Training history saved to: {history_path}")

    return history_path


def print_header(experiment_name: str, arch: str, config_name: str):
    """Print a formatted experiment header."""
    print("=" * 70)
    print(f"{experiment_name}: {arch.upper()} + {config_name.upper()}")
    print("=" * 70)


def print_results(eval_results: Dict[str, Any]):
    """Print physical-domain evaluation results."""
    phys = eval_results["physical"]

    print(f"\n{'─' * 50}")
    print("Test results - physical domain (per-earthquake nRMSE)")
    print(f"{'─' * 50}")
    print(
        f"  Field nRMSE:         {phys.field_nrmse_mean:.4f} ± {phys.field_nrmse_std:.4f}"
    )
    print(
        f"  Log Spectral nRMSE:  {phys.log_spectral_nrmse_mean:.4f} ± {phys.log_spectral_nrmse_std:.4f}"
    )
    print(
        f"  Barron Norm Error:   {phys.barron_norm_error_mean:.4f} ± {phys.barron_norm_error_std:.4f}"
    )
    print(f"{'─' * 50}")
