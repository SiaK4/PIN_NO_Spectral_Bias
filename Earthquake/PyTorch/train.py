#!/usr/bin/env python3
"""
Training script for the full ablation study.

Configs are defined in CONFIGS below. Data is physical-space CDON,
LR is 1e-3 with cosine decay to 0.

Usage:
    python train.py --index 0 --device cuda
"""

import argparse
import copy
import sys
from pathlib import Path
from typing import Optional

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from configs.loss_config import LOSS_CONFIG_MAP, LossConfig
from configs.model_configs import MODEL_CONFIGS
from src.core.evaluation.constants import BATCH_SIZE_SEQUENCE, CNO_INTERNAL_SIZE
from src.core.models.model_factory import create_model
from src.core.training.experiment_utils import (
    create_dataloaders,
    create_trainer,
    evaluate_and_get_metrics,
    get_run_name,
    load_best_model,
    prepare_model,
    print_header,
    print_results,
    save_results,
    save_training_history,
)

LOSS_CONFIGS = LOSS_CONFIG_MAP

# 32 configurations: (arch, activation, causal_mode, optimizer, loss)
# causal_mode: 'causal' or 'seq' for DeepONet/DeepOKAN, None for FNO/CNO
CONFIGS = [
    # DeepONet SIREN causal (indices 0-3)
    ("deeponet", "siren", "causal", "soap", "baseline"),
    ("deeponet", "siren", "causal", "soap", "bsp"),
    ("deeponet", "siren", "causal", "adam", "baseline"),
    ("deeponet", "siren", "causal", "adam", "bsp"),
    # DeepONet SIREN seq (indices 4-7)
    ("deeponet", "siren", "seq", "soap", "baseline"),
    ("deeponet", "siren", "seq", "soap", "bsp"),
    ("deeponet", "siren", "seq", "adam", "baseline"),
    ("deeponet", "siren", "seq", "adam", "bsp"),
    # DeepONet Tanh causal (indices 8-11)
    ("deeponet", "tanh", "causal", "soap", "baseline"),
    ("deeponet", "tanh", "causal", "soap", "bsp"),
    ("deeponet", "tanh", "causal", "adam", "baseline"),
    ("deeponet", "tanh", "causal", "adam", "bsp"),
    # DeepONet Tanh seq (indices 12-15)
    ("deeponet", "tanh", "seq", "soap", "baseline"),
    ("deeponet", "tanh", "seq", "soap", "bsp"),
    ("deeponet", "tanh", "seq", "adam", "baseline"),
    ("deeponet", "tanh", "seq", "adam", "bsp"),
    # DeepOKAN causal (indices 16-19)
    ("deepokan", None, "causal", "soap", "baseline"),
    ("deepokan", None, "causal", "soap", "bsp"),
    ("deepokan", None, "causal", "adam", "baseline"),
    ("deepokan", None, "causal", "adam", "bsp"),
    # DeepOKAN seq (indices 20-23)
    ("deepokan", None, "seq", "soap", "baseline"),
    ("deepokan", None, "seq", "soap", "bsp"),
    ("deepokan", None, "seq", "adam", "baseline"),
    ("deepokan", None, "seq", "adam", "bsp"),
    # FNO (indices 24-27)
    ("fno", None, None, "soap", "baseline"),
    ("fno", None, None, "soap", "bsp"),
    ("fno", None, None, "adam", "baseline"),
    ("fno", None, None, "adam", "bsp"),
    # CNO (indices 28-31)
    ("cno", None, None, "soap", "baseline"),
    ("cno", None, None, "soap", "bsp"),
    ("cno", None, None, "adam", "baseline"),
    ("cno", None, None, "adam", "bsp"),
]


def train_single_config(
    arch: str,
    optimizer: str,
    loss_name: str,
    activation: Optional[str] = None,
    causal_mode: Optional[str] = None,
    device: str = "cuda",
    seed: int = 42,
    checkpoint_suffix: str = "",
):
    """Train a single architecture + optimizer + loss configuration."""
    max_time = 1080

    is_deeponet_like = arch in ["deeponet", "deepokan"]
    if causal_mode is not None:
        use_causal = causal_mode == "causal"
    else:
        use_causal = False  # FNO/CNO don't support causal

    # Equal wall-clock budget per job.
    epochs = 500 if use_causal else 1500

    checkpoint_dir = (
        project_root / "checkpoints" / f"soap_vs_adam_ablation{checkpoint_suffix}"
    )
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    run_name = get_run_name(arch, optimizer, loss_name, activation, causal_mode)

    config_str = f"{optimizer} + {loss_name}"
    if is_deeponet_like:
        config_str = f"{'causal' if use_causal else 'sequence'} + {config_str}"
    if arch == "deeponet" and activation:
        config_str = f"{activation} + {config_str}"
    print_header("SOAP vs Adam ablation", arch, config_str)
    print(f"Checkpoint dir: {checkpoint_dir}")
    print(f"Epochs: {epochs}")
    print(f"Device: {device}")
    print(f"Optimizer: {optimizer.upper()}")
    print(f"Causal: {'ENABLED' if use_causal else 'DISABLED'}")

    print("\n1. Creating model...")
    model_config = copy.deepcopy(MODEL_CONFIGS[arch])

    if arch == "deeponet" and activation:
        if activation == "tanh":
            model_config["branch_activation"] = "tanh"
            model_config["trunk_activation"] = "tanh"
        elif activation == "siren":
            model_config["branch_activation"] = "tanh"
            model_config["trunk_activation"] = "siren"

    model = create_model(arch, config=model_config)
    model = prepare_model(model, device)
    print(f"   Architecture: {arch}")
    if arch == "deeponet" and activation:
        print(f"   Branch activation: {model_config['branch_activation']}")
        print(f"   Trunk activation: {model_config['trunk_activation']}")
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")

    is_cno = arch == "cno"

    base_loss_config = LOSS_CONFIGS[loss_name]
    loss_config = LossConfig(
        loss_type=base_loss_config.loss_type,
        loss_params=copy.deepcopy(base_loss_config.loss_params),
        description=base_loss_config.description,
    )

    if loss_name != "baseline" and is_cno:
        loss_config.loss_params["signal_length"] = CNO_INTERNAL_SIZE

    print(f"   Loss: {loss_name}")
    print(f"   Description: {loss_config.description}")
    if loss_name != "baseline":
        print(f"   w_bsp: {loss_config.loss_params.get('w_bsp', 'not set')}")

    print("\n2. Loading physical-space data...")
    batch_size = BATCH_SIZE_SEQUENCE
    is_baseline = loss_name == "baseline"

    loaders = create_dataloaders(
        arch,
        batch_size=batch_size,
        is_baseline=is_baseline,
        use_causal=use_causal,
    )

    if loaders.get("is_deeponet", False) and use_causal:
        print(
            f"   Per-timestep samples: {loaders.get('per_timestep_samples', 'unknown')}"
        )
        if not is_baseline:
            print(f"   Sequence samples: {loaders.get('sequence_samples', 'unknown')}")
        print(
            f"   Mode: {'Flat per-timestep (baseline)' if is_baseline else 'Paired sampling (BSP)'}"
        )
    elif loaders.get("is_deeponet", False) and not use_causal:
        print(f"   Sequence samples: {loaders['sequence_samples']}")
        print("   Mode: Sequence-only (non-causal, like FNO/CNO)")
    else:
        print(f"   Sequence samples: {loaders['sequence_samples']}")
        print("   Mode: Sequence-only (FNO/CNO style)")

    trainer = create_trainer(
        model=model,
        loaders=loaders,
        loss_config=loss_config,
        experiment_name=run_name,
        checkpoint_dir=checkpoint_dir,
        epochs=epochs,
        device=device,
        arch=arch,
        optimizer_override=optimizer,
        use_causal=use_causal,
        max_training_time=max_time if max_time > 0 else None,
        seed=seed,
    )
    print(f"   Optimizer: {optimizer.upper()}")
    print(f"   Epochs: {epochs}")
    if max_time > 0:
        print(f"   Max time: {max_time/60:.1f} min")

    print("\n3. Training...")
    trainer.train()

    save_training_history(
        trainer,
        checkpoint_dir,
        run_name,
        use_causal=use_causal,
        extra_fields={
            "metric_domain": "physical",
            "optimizer": optimizer,
        },
    )

    eval_use_causal = is_deeponet_like and use_causal

    print("\n4. Evaluating on test set...")
    model = load_best_model(model, checkpoint_dir, run_name, device)
    eval_results = evaluate_and_get_metrics(
        model,
        loaders["test"],
        device,
        use_causal=eval_use_causal,
        arch=arch,
    )

    print_results(eval_results)

    # Extract physical domain metrics
    phys = eval_results["physical"]

    results = {
        "arch": arch,
        "optimizer": optimizer,
        "loss": loss_name,
        "causal": use_causal,
        "activation": activation if arch == "deeponet" else None,
        "epochs": epochs,
        "seed": seed,
        "use_causal": use_causal,
        "settings": {
            "w_bsp": loss_config.loss_params.get("w_bsp"),
        },
        "field_nrmse_mean": float(phys.field_nrmse_mean),
        "field_nrmse_std": float(phys.field_nrmse_std),
        "log_spectral_nrmse_mean": float(phys.log_spectral_nrmse_mean),
        "log_spectral_nrmse_std": float(phys.log_spectral_nrmse_std),
        "barron_norm_mean": float(phys.barron_norm_error_mean),
        "barron_norm_std": float(phys.barron_norm_error_std),
    }

    results_path = save_results(results, checkpoint_dir, run_name)
    print(f"\nResults saved to: {results_path}")
    print("=" * 70)

    return results


def main():
    parser = argparse.ArgumentParser(description="Train SOAP vs Adam ablation")
    parser.add_argument(
        "--index",
        type=int,
        required=True,
        help=f"Configuration index (0-{len(CONFIGS)-1})",
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="Device to use for training"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--checkpoint-suffix",
        type=str,
        default="",
        help="Suffix for checkpoint directory (e.g., _seed42)",
    )

    args = parser.parse_args()

    if args.index < 0 or args.index >= len(CONFIGS):
        print(f"ERROR: Index {args.index} out of range [0, {len(CONFIGS)-1}]")
        sys.exit(1)

    arch, activation, causal_mode, optimizer, loss_name = CONFIGS[args.index]
    print(
        f"Index {args.index}: arch={arch}, activation={activation}, "
        f"causal={causal_mode}, optimizer={optimizer}, loss={loss_name}, "
        f"seed={args.seed}"
    )

    train_single_config(
        arch=arch,
        optimizer=optimizer,
        loss_name=loss_name,
        activation=activation,
        causal_mode=causal_mode,
        device=args.device,
        seed=args.seed,
        checkpoint_suffix=args.checkpoint_suffix,
    )


if __name__ == "__main__":
    main()
