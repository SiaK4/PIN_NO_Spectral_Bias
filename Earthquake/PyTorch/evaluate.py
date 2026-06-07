#!/usr/bin/env python3
"""
Evaluation script for the SOAP vs Adam optimizer ablation.

Reads eval_results.json files and produces a markdown summary table plus
spectral, 4-panel, training-curve, and combined derivative/arch-comparison plots.

Usage:
    python evaluate.py
    python evaluate.py --checkpoint-suffix _seed42 --output-suffix _seed42
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.core.data_processing.cdon_dataset import CDONDataset
from src.core.evaluation.constants import (BATCH_SIZE_SEQUENCE,
                                           CNO_INTERNAL_SIZE,
                                           SIGNAL_LENGTH_CDON)
from src.core.evaluation.metrics import compute_derivative_metrics
from src.core.training.experiment_utils import (create_test_dataloader,
                                                evaluate_on_test_set,
                                                get_run_name,
                                                load_trained_model)
from src.core.visualization import (ACTIVATION_LABELS, ARCH_LABELS,
                                    OPTIMIZER_COLORS, OPTIMIZER_LABELS,
                                    plot_4panel_comparison,
                                    plot_arch_loss_comparison_grid,
                                    plot_combined_derivative_grid,
                                    plot_energy_spectrum_comparison,
                                    plot_training_curves,
                                    select_visualization_samples)

# Arch-variants as (arch, activation, causal_mode) tuples
# causal_mode: 'causal' or 'seq' for DeepONet/DeepOKAN, None for FNO/CNO
ARCH_VARIANTS = [
    ("deeponet", "siren", "causal"),
    ("deeponet", "siren", "seq"),
    ("deeponet", "tanh", "causal"),
    ("deeponet", "tanh", "seq"),
    ("deepokan", None, "causal"),
    ("deepokan", None, "seq"),
    ("fno", None, None),
    ("cno", None, None),
]

OPTIMIZERS = ["soap", "adam"]
LOSSES = ["baseline", "bsp"]

LOSS_LABELS = {
    "baseline": "Baseline (L2 Norm)",
    "bsp": "Log-BSP (w=1.0)",
}

CAUSAL_LABELS = {
    "causal": "Causal (per-timestep)",
    "seq": "Sequence (non-causal)",
}


def get_variant_label(
    arch: str, activation: Optional[str] = None, causal_mode: Optional[str] = None
) -> str:
    """Get display label for an arch-variant."""
    label = ARCH_LABELS[arch]
    if arch == "deeponet" and activation:
        label = f"{label} ({ACTIVATION_LABELS[activation]})"
    if causal_mode:
        label = f"{label} [{CAUSAL_LABELS[causal_mode].split()[0]}]"
    return label

def load_eval_results(
    arch: str,
    optimizer: str,
    loss: str,
    checkpoint_dir: Path,
    activation: Optional[str] = None,
    causal_mode: Optional[str] = None,
) -> Optional[Dict]:
    """Load evaluation results from JSON file."""
    run_name = get_run_name(arch, optimizer, loss, activation, causal_mode)
    results_path = checkpoint_dir / run_name / "eval_results.json"

    if results_path.exists():
        with open(results_path) as f:
            return json.load(f)
    return None


def get_training_history_path(
    arch: str,
    optimizer: str,
    loss: str,
    checkpoint_dir: Path,
    activation: Optional[str] = None,
    causal_mode: Optional[str] = None,
) -> Optional[Path]:
    """Get path to training history file."""
    run_name = get_run_name(arch, optimizer, loss, activation, causal_mode)
    history_path = checkpoint_dir / run_name / "training_history.npz"
    return history_path if history_path.exists() else None


def generate_summary_table(output_dir: Path, checkpoint_dir: Path):
    """Generate summary table in markdown format."""
    print("\n" + "=" * 70)
    print("Generating Summary Table...")
    print("=" * 70)

    all_results = []

    for arch, activation, causal_mode in ARCH_VARIANTS:
        for optimizer in OPTIMIZERS:
            for loss in LOSSES:
                results = load_eval_results(
                    arch,
                    optimizer,
                    loss,
                    checkpoint_dir,
                    activation=activation,
                    causal_mode=causal_mode,
                )
                if results:
                    all_results.append(
                        {
                            "arch": arch,
                            "activation": activation,
                            "causal_mode": causal_mode,
                            "optimizer": optimizer,
                            "loss": loss,
                            "field_nrmse_mean": results.get(
                                "field_nrmse_mean", float("nan")
                            ),
                            "field_nrmse_std": results.get(
                                "field_nrmse_std", float("nan")
                            ),
                            "log_spectral_nrmse_mean": results.get(
                                "log_spectral_nrmse_mean", float("nan")
                            ),
                            "log_spectral_nrmse_std": results.get(
                                "log_spectral_nrmse_std", float("nan")
                            ),
                            "barron_norm_mean": results.get(
                                "barron_norm_mean", float("nan")
                            ),
                            "barron_norm_std": results.get(
                                "barron_norm_std", float("nan")
                            ),
                            "velocity_nrmse_mean": results.get(
                                "velocity_nrmse_mean", float("nan")
                            ),
                            "velocity_nrmse_std": results.get(
                                "velocity_nrmse_std", float("nan")
                            ),
                            "acceleration_nrmse_mean": results.get(
                                "acceleration_nrmse_mean", float("nan")
                            ),
                            "acceleration_nrmse_std": results.get(
                                "acceleration_nrmse_std", float("nan")
                            ),
                        }
                    )
                else:
                    all_results.append(
                        {
                            "arch": arch,
                            "activation": activation,
                            "causal_mode": causal_mode,
                            "optimizer": optimizer,
                            "loss": loss,
                            "field_nrmse_mean": float("nan"),
                            "field_nrmse_std": float("nan"),
                            "log_spectral_nrmse_mean": float("nan"),
                            "log_spectral_nrmse_std": float("nan"),
                            "barron_norm_mean": float("nan"),
                            "barron_norm_std": float("nan"),
                            "velocity_nrmse_mean": float("nan"),
                            "velocity_nrmse_std": float("nan"),
                            "acceleration_nrmse_mean": float("nan"),
                            "acceleration_nrmse_std": float("nan"),
                        }
                    )

    # Generate markdown
    md_lines = [
        "# SOAP vs Adam Optimizer Ablation Results",
        "",
        "## Configuration",
        "",
        "- **Architectures**: DeepONet (SIREN + Tanh), DeepOKAN, FNO, CNO",
        "- **Causal Modes**: Causal (per-timestep), Sequence (non-causal) for DeepONet/DeepOKAN",
        "- **Optimizers**: SOAP (lr=1e-3), Adam (lr=1e-3)",
        "- **Loss Variants**: Baseline (L2 Norm), Log-BSP (w=1.0)",
        "- **Training**: 18 min max time, physical-space data",
        "",
        "## Summary Tables",
        "",
    ]

    # Generate table for each loss variant
    for loss in LOSSES:
        # Field nRMSE table
        md_lines.extend(
            [
                f"### {LOSS_LABELS[loss]} - Field nRMSE (mean +/- std)",
                "",
                "| Configuration | "
                + " | ".join(OPTIMIZER_LABELS[o] for o in OPTIMIZERS)
                + " |",
                "|" + "-|" * (len(OPTIMIZERS) + 1),
            ]
        )

        for arch, activation, causal_mode in ARCH_VARIANTS:
            label = get_variant_label(arch, activation, causal_mode)
            row = f"| **{label}** |"
            for optimizer in OPTIMIZERS:
                result = next(
                    (
                        r
                        for r in all_results
                        if r["arch"] == arch
                        and r["optimizer"] == optimizer
                        and r["loss"] == loss
                        and r.get("activation") == activation
                        and r.get("causal_mode") == causal_mode
                    ),
                    None,
                )
                if result and not np.isnan(result["field_nrmse_mean"]):
                    row += f" {result['field_nrmse_mean']:.4f} +/- {result['field_nrmse_std']:.4f} |"
                else:
                    row += " N/A |"
            md_lines.append(row)

        # Log Spectral nRMSE table
        md_lines.extend(
            [
                "",
                f"### {LOSS_LABELS[loss]} - Log Spectral nRMSE (mean +/- std)",
                "",
                "| Configuration | "
                + " | ".join(OPTIMIZER_LABELS[o] for o in OPTIMIZERS)
                + " |",
                "|" + "-|" * (len(OPTIMIZERS) + 1),
            ]
        )

        for arch, activation, causal_mode in ARCH_VARIANTS:
            label = get_variant_label(arch, activation, causal_mode)
            row = f"| **{label}** |"
            for optimizer in OPTIMIZERS:
                result = next(
                    (
                        r
                        for r in all_results
                        if r["arch"] == arch
                        and r["optimizer"] == optimizer
                        and r["loss"] == loss
                        and r.get("activation") == activation
                        and r.get("causal_mode") == causal_mode
                    ),
                    None,
                )
                if result and not np.isnan(result["log_spectral_nrmse_mean"]):
                    row += f" {result['log_spectral_nrmse_mean']:.4f} +/- {result['log_spectral_nrmse_std']:.4f} |"
                else:
                    row += " N/A |"
            md_lines.append(row)

        # Barron Norm table
        md_lines.extend(
            [
                "",
                f"### {LOSS_LABELS[loss]} - Barron Norm Error (mean +/- std)",
                "",
                "| Configuration | "
                + " | ".join(OPTIMIZER_LABELS[o] for o in OPTIMIZERS)
                + " |",
                "|" + "-|" * (len(OPTIMIZERS) + 1),
            ]
        )

        for arch, activation, causal_mode in ARCH_VARIANTS:
            label = get_variant_label(arch, activation, causal_mode)
            row = f"| **{label}** |"
            for optimizer in OPTIMIZERS:
                result = next(
                    (
                        r
                        for r in all_results
                        if r["arch"] == arch
                        and r["optimizer"] == optimizer
                        and r["loss"] == loss
                        and r.get("activation") == activation
                        and r.get("causal_mode") == causal_mode
                    ),
                    None,
                )
                if result and not np.isnan(
                    result.get("barron_norm_mean", float("nan"))
                ):
                    row += f" {result['barron_norm_mean']:.4f} +/- {result['barron_norm_std']:.4f} |"
                else:
                    row += " N/A |"
            md_lines.append(row)

        # Velocity nRMSE table
        md_lines.extend(
            [
                "",
                f"### {LOSS_LABELS[loss]} - Velocity nRMSE (mean +/- std)",
                "",
                "| Configuration | "
                + " | ".join(OPTIMIZER_LABELS[o] for o in OPTIMIZERS)
                + " |",
                "|" + "-|" * (len(OPTIMIZERS) + 1),
            ]
        )

        for arch, activation, causal_mode in ARCH_VARIANTS:
            label = get_variant_label(arch, activation, causal_mode)
            row = f"| **{label}** |"
            for optimizer in OPTIMIZERS:
                result = next(
                    (
                        r
                        for r in all_results
                        if r["arch"] == arch
                        and r["optimizer"] == optimizer
                        and r["loss"] == loss
                        and r.get("activation") == activation
                        and r.get("causal_mode") == causal_mode
                    ),
                    None,
                )
                if result and not np.isnan(
                    result.get("velocity_nrmse_mean", float("nan"))
                ):
                    row += f" {result['velocity_nrmse_mean']:.4f} +/- {result['velocity_nrmse_std']:.4f} |"
                else:
                    row += " N/A |"
            md_lines.append(row)

        # Acceleration nRMSE table
        md_lines.extend(
            [
                "",
                f"### {LOSS_LABELS[loss]} - Acceleration nRMSE (mean +/- std)",
                "",
                "| Configuration | "
                + " | ".join(OPTIMIZER_LABELS[o] for o in OPTIMIZERS)
                + " |",
                "|" + "-|" * (len(OPTIMIZERS) + 1),
            ]
        )

        for arch, activation, causal_mode in ARCH_VARIANTS:
            label = get_variant_label(arch, activation, causal_mode)
            row = f"| **{label}** |"
            for optimizer in OPTIMIZERS:
                result = next(
                    (
                        r
                        for r in all_results
                        if r["arch"] == arch
                        and r["optimizer"] == optimizer
                        and r["loss"] == loss
                        and r.get("activation") == activation
                        and r.get("causal_mode") == causal_mode
                    ),
                    None,
                )
                if result and not np.isnan(
                    result.get("acceleration_nrmse_mean", float("nan"))
                ):
                    row += f" {result['acceleration_nrmse_mean']:.4f} +/- {result['acceleration_nrmse_std']:.4f} |"
                else:
                    row += " N/A |"
            md_lines.append(row)

        md_lines.append("")

    # Find best configurations
    valid_results = [r for r in all_results if not np.isnan(r["field_nrmse_mean"])]
    if valid_results:
        best_l2 = min(valid_results, key=lambda r: r["field_nrmse_mean"])
        best_spec = min(valid_results, key=lambda r: r["log_spectral_nrmse_mean"])
        valid_barron = [
            r
            for r in valid_results
            if not np.isnan(r.get("barron_norm_mean", float("nan")))
        ]
        best_barron = (
            min(valid_barron, key=lambda r: r["barron_norm_mean"])
            if valid_barron
            else None
        )
        valid_velocity = [
            r
            for r in valid_results
            if not np.isnan(r.get("velocity_nrmse_mean", float("nan")))
        ]
        best_velocity = (
            min(valid_velocity, key=lambda r: r["velocity_nrmse_mean"])
            if valid_velocity
            else None
        )
        valid_accel = [
            r
            for r in valid_results
            if not np.isnan(r.get("acceleration_nrmse_mean", float("nan")))
        ]
        best_accel = (
            min(valid_accel, key=lambda r: r["acceleration_nrmse_mean"])
            if valid_accel
            else None
        )

        md_lines.extend(
            [
                "## Best Configurations",
                "",
                f"- **Best Field nRMSE**: {get_variant_label(best_l2['arch'], best_l2.get('activation'), best_l2.get('causal_mode'))} + "
                f"{OPTIMIZER_LABELS[best_l2['optimizer']]} + "
                f"{LOSS_LABELS[best_l2['loss']]} ({best_l2['field_nrmse_mean']:.4f})",
                f"- **Best Log Spectral nRMSE**: {get_variant_label(best_spec['arch'], best_spec.get('activation'), best_spec.get('causal_mode'))} + "
                f"{OPTIMIZER_LABELS[best_spec['optimizer']]} + "
                f"{LOSS_LABELS[best_spec['loss']]} ({best_spec['log_spectral_nrmse_mean']:.4f})",
            ]
        )
        if best_barron:
            md_lines.append(
                f"- **Best Barron Norm**: {get_variant_label(best_barron['arch'], best_barron.get('activation'), best_barron.get('causal_mode'))} + "
                f"{OPTIMIZER_LABELS[best_barron['optimizer']]} + "
                f"{LOSS_LABELS[best_barron['loss']]} ({best_barron['barron_norm_mean']:.4f})"
            )
        if best_velocity:
            md_lines.append(
                f"- **Best Velocity nRMSE**: {get_variant_label(best_velocity['arch'], best_velocity.get('activation'), best_velocity.get('causal_mode'))} + "
                f"{OPTIMIZER_LABELS[best_velocity['optimizer']]} + "
                f"{LOSS_LABELS[best_velocity['loss']]} ({best_velocity['velocity_nrmse_mean']:.4f})"
            )
        if best_accel:
            md_lines.append(
                f"- **Best Acceleration nRMSE**: {get_variant_label(best_accel['arch'], best_accel.get('activation'), best_accel.get('causal_mode'))} + "
                f"{OPTIMIZER_LABELS[best_accel['optimizer']]} + "
                f"{LOSS_LABELS[best_accel['loss']]} ({best_accel['acceleration_nrmse_mean']:.4f})"
            )

    # Save markdown
    md_path = output_dir / "summary_table.md"
    with open(md_path, "w") as f:
        f.write("\n".join(md_lines))
    print(f"  Summary table saved to: {md_path}")

    # Save JSON for programmatic access
    json_path = output_dir / "all_results.json"
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"  JSON results saved to: {json_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate SOAP vs Adam ablation")
    parser.add_argument(
        "--checkpoint-suffix",
        type=str,
        default="",
        help='Suffix for checkpoint dir (e.g., "_seed42")',
    )
    parser.add_argument(
        "--output-suffix",
        type=str,
        default="",
        help='Suffix for output dir (e.g., "_seed42")',
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Directories with optional suffix
    checkpoint_dir = (
        project_root / "checkpoints" / f"soap_vs_adam_ablation{args.checkpoint_suffix}"
    )
    output_dir = project_root / "results" / f"soap_vs_adam_ablation{args.output_suffix}"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Checkpoint dir: {checkpoint_dir}")
    print(f"Output dir: {output_dir}")

    # Reads from eval_results.json files
    generate_summary_table(output_dir, checkpoint_dir)

    print("\n" + "=" * 70)
    print("Loading models and generating predictions...")
    print("=" * 70)

    data_dir = project_root / "CDONData"

    all_soap_predictions = {}
    all_targets = {}

    for arch, activation, causal_mode in ARCH_VARIANTS:
        is_deeponet_like = arch in ["deeponet", "deepokan"]
        arch_label = get_variant_label(arch, activation, causal_mode)

        # CNO requires 4096 signal length (interpolated from 4000)
        is_cno = arch == "cno"
        target_signal_length = CNO_INTERNAL_SIZE if is_cno else None
        print(f"\n  Loading test data for {arch_label}...")

        test_dataset = CDONDataset(
            data_dir=str(data_dir),
            split="test",
            mode="sequence",
            signal_length=SIGNAL_LENGTH_CDON,
            target_signal_length=target_signal_length,
        )
        test_loader = create_test_dataloader(
            test_dataset, batch_size=BATCH_SIZE_SEQUENCE
        )

        for loss in LOSSES:
            print(f"\n  Processing {arch_label} - {LOSS_LABELS[loss]}...")

            optimizer_predictions = {}
            targets = None
            inputs = None

            for optimizer in OPTIMIZERS:
                model = load_trained_model(
                    arch,
                    optimizer,
                    loss,
                    checkpoint_dir,
                    device=device,
                    activation=activation,
                    causal_mode=causal_mode,
                )

                if model is not None:
                    # Only DeepONet and DeepOKAN can use causal
                    use_causal = (
                        (causal_mode == "causal") if is_deeponet_like else False
                    )

                    _, _, preds_denorm, tgts_denorm, inps_denorm, _ = (
                        evaluate_on_test_set(
                            model,
                            test_loader,
                            device,
                            use_causal=use_causal,
                            return_inputs=True,
                            arch=arch,
                        )
                    )

                    preds_np = preds_denorm.numpy()
                    optimizer_predictions[optimizer] = preds_np

                    if targets is None:
                        targets = tgts_denorm.numpy()
                        inputs = inps_denorm.numpy()

                    run_name = get_run_name(
                        arch, optimizer, loss, activation, causal_mode
                    )
                    print(f"    Loaded {run_name} (use_causal={use_causal})")

                    deriv = compute_derivative_metrics(preds_np, targets, dt=0.02)
                    eval_path = checkpoint_dir / run_name / "eval_results.json"
                    if eval_path.exists():
                        with open(eval_path) as f:
                            existing = json.load(f)
                        existing.update(deriv)
                        with open(eval_path, "w") as f:
                            json.dump(existing, f, indent=2)
                        print(
                            f"    Updated {run_name}/eval_results.json with derivative metrics"
                        )

                    if optimizer == "soap":
                        include_in_combined = (
                            arch != "deeponet" or activation == "siren"
                        )
                        include_in_combined = include_in_combined and (
                            not is_deeponet_like or causal_mode == "causal"
                        )
                        if include_in_combined:
                            if arch not in all_soap_predictions:
                                all_soap_predictions[arch] = {}
                            all_soap_predictions[arch][loss] = preds_np
                            if arch not in all_targets:
                                all_targets[arch] = tgts_denorm.numpy()
                else:
                    run_name = get_run_name(
                        arch, optimizer, loss, activation, causal_mode
                    )
                    print(f"    SKIPPED {run_name} (not found)")

            if len(optimizer_predictions) == 0:
                print("    No models found, skipping visualizations")
                continue

            if arch == "deeponet" and activation:
                file_prefix = f"{arch}_{activation}"
            else:
                file_prefix = arch
            if causal_mode:
                file_prefix = f"{file_prefix}_{causal_mode}"
            file_prefix = f"{file_prefix}_{loss}"

            print("  Generating visualizations...")

            # Spectral comparison with both optimizers overlaid
            plot_energy_spectrum_comparison(
                predictions=optimizer_predictions,
                targets=targets,
                output_path=str(output_dir / f"{file_prefix}_spectrum_comparison.png"),
                title=f"{arch_label} {LOSS_LABELS[loss]} - Optimizer Comparison",
                colors=OPTIMIZER_COLORS,
                labels=OPTIMIZER_LABELS,
            )
            print("    Generated spectrum plot")

            # 4-panel comparison plots for best/worst samples
            baseline_key = (
                "soap"
                if "soap" in optimizer_predictions
                else list(optimizer_predictions.keys())[0]
            )
            baseline_preds = optimizer_predictions[baseline_key]
            sample_indices, sample_labels = select_visualization_samples(
                baseline_preds, targets, n_samples=2
            )
            print(f"    Selected samples: {dict(zip(sample_indices, sample_labels))}")

            for sample_idx, sample_label in zip(sample_indices, sample_labels):
                if sample_idx < targets.shape[0]:
                    label_clean = sample_label.split(" (")[0].lower().replace(" ", "_")
                    plot_4panel_comparison(
                        predictions=optimizer_predictions,
                        target=targets,
                        input_signal=inputs,
                        sample_idx=sample_idx,
                        output_path=str(
                            output_dir / f"{file_prefix}_4panel_{label_clean}.png"
                        ),
                        title=f"{arch_label} {LOSS_LABELS[loss]} - {sample_label}",
                        colors=OPTIMIZER_COLORS,
                        labels=OPTIMIZER_LABELS,
                    )
                    print(f"    Generated 4-panel plot ({sample_label})")

            # Training curves with both optimizers overlaid
            history_paths = {}
            for optimizer in OPTIMIZERS:
                path = get_training_history_path(
                    arch,
                    optimizer,
                    loss,
                    checkpoint_dir,
                    activation=activation,
                    causal_mode=causal_mode,
                )
                if path:
                    history_paths[optimizer] = str(path)

            if history_paths:
                plot_training_curves(
                    history_paths=history_paths,
                    output_path=str(output_dir / f"{file_prefix}_training_curves.png"),
                    title=f"{arch_label} {LOSS_LABELS[loss]} - Training Curves",
                    colors=OPTIMIZER_COLORS,
                    labels=OPTIMIZER_LABELS,
                )
                print("    Generated training curves")

    if all_soap_predictions and "deeponet" in all_targets:
        print(
            "\n  Generating combined plots (SOAP optimizer, causal mode for DeepONet/DeepOKAN)..."
        )
        combined_targets = all_targets["deeponet"]

        plot_combined_derivative_grid(
            all_predictions=all_soap_predictions,
            targets=combined_targets,
            output_path=str(output_dir / "combined_best_samples.png"),
            mode="best",
            title="Combined Derivative Analysis (SOAP) - Best Sample per Model",
            arch_labels=ARCH_LABELS,
            loss_labels=LOSS_LABELS,
        )
        print("    Generated combined_best_samples.png")

        plot_combined_derivative_grid(
            all_predictions=all_soap_predictions,
            targets=combined_targets,
            output_path=str(output_dir / "combined_worst_samples.png"),
            mode="worst",
            title="Combined Derivative Analysis (SOAP) - Worst Sample per Model",
            arch_labels=ARCH_LABELS,
            loss_labels=LOSS_LABELS,
        )
        print("    Generated combined_worst_samples.png")

        plot_arch_loss_comparison_grid(
            all_predictions=all_soap_predictions,
            targets=combined_targets,
            output_path=str(output_dir / "arch_comparison_best.png"),
            mode="best",
            title="Architecture Comparison (SOAP) - Best L2 Sample per Model",
            arch_labels=ARCH_LABELS,
        )
        print("    Generated arch_comparison_best.png")

        plot_arch_loss_comparison_grid(
            all_predictions=all_soap_predictions,
            targets=combined_targets,
            output_path=str(output_dir / "arch_comparison_worst.png"),
            mode="worst",
            title="Architecture Comparison (SOAP) - Worst L2 Sample per Model",
            arch_labels=ARCH_LABELS,
        )
        print("    Generated arch_comparison_worst.png")

    # Regenerate summary table now that derivative metrics have been computed
    generate_summary_table(output_dir, checkpoint_dir)

    print(f"\n{'=' * 70}")
    print(f"All outputs saved to: {output_dir}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
