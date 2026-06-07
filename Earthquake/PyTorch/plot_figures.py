#!/usr/bin/env python3
"""
Create illustrative comparison figures.

Usage:
    python plot_figures.py
    python plot_figures.py --seeds 1,2,3 --output-dir results/
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.core.data_processing.cdon_dataset import CDONDataset
from src.core.evaluation.constants import (
    BATCH_SIZE_SEQUENCE,
    CNO_INTERNAL_SIZE,
    SIGNAL_LENGTH_CDON,
)
from src.core.evaluation.utils import postprocess_cno_predictions
from src.core.training.experiment_utils import (
    create_test_dataloader,
    evaluate_on_test_set,
    load_trained_model,
)
from src.core.visualization import (
    ACTIVATION_COLORS,
    ARCH_COLORS,
    ARCH_LABELS,
    LOSS_COLORS_DEFAULT,
    OPTIMIZER_COLORS,
    plot_compact_impact_grid,
    plot_fixed_sample_comparison_grid,
)
from src.core.visualization.plotting_utils import _extract_sample

# Fixed example shown in every figure
DISPLAY_SAMPLE = 12
DISPLAY_SEED = 1


def get_predictions(
    model: torch.nn.Module, test_loader, device: str, use_causal: bool, arch: str
) -> Tuple[np.ndarray, np.ndarray]:
    """Run inference and return (predictions, targets) as numpy arrays."""
    _, _, preds_physical, targets_physical, _ = evaluate_on_test_set(
        model,
        test_loader,
        device,
        use_causal=use_causal,
        return_inputs=False,
        arch=arch,
    )
    return preds_physical.numpy(), targets_physical.numpy()


def define_comparisons() -> List[Dict]:
    """Define the comparison configurations (one figure each).

    Each comparison specifies:
    - name: identifier for output files
    - title: figure title
    - group_a_label / group_b_label: column group headers
    - archs: list of (arch, activation, causal_mode) tuples
    - variant_a_spec / variant_b_spec: (optimizer, loss) for each column group
    """
    standard_archs = [
        ("deeponet", "siren", "causal"),
        ("deepokan", None, "causal"),
        ("fno", None, None),
        ("cno", None, None),
    ]

    return [
        # 1. BSP > Baseline
        {
            "name": "bsp_improvement",
            "title": None,
            "group_a_label": "Baseline",
            "group_b_label": "BSP",
            "archs": standard_archs,
            "variant_a_spec": ("soap", "baseline"),
            "variant_b_spec": ("soap", "bsp"),
        },
        # 2a. SOAP > Adam (baseline loss)
        {
            "name": "optimizer_baseline_improvement",
            "title": None,
            "group_a_label": "Adam",
            "group_b_label": "SOAP",
            "archs": standard_archs,
            "variant_a_spec": ("adam", "baseline"),
            "variant_b_spec": ("soap", "baseline"),
        },
        # 2b. SOAP > Adam (BSP loss) not plotted here
        # 3. SIREN > Tanh (DeepONet only, baseline + BSP rows)
        {
            "name": "activation_comparison",
            "title": None,
            "group_a_label": "Tanh",
            "group_b_label": "SIREN",
            "archs": [
                ("deeponet", "tanh", "causal"),
                ("deeponet", "siren", "causal"),
            ],
            "variant_a_spec": ("soap", "baseline"),
            "variant_b_spec": ("soap", "baseline"),
            # Special: rows are (baseline, BSP) not (arch1, arch2)
            "special_mode": "activation",
        },
        # 4. Causal > Non-causal
        {
            "name": "causal_necessity",
            "title": None,
            "group_a_label": "Non-Causal",
            "group_b_label": "Causal",
            "archs": [
                ("deeponet", "siren", "seq"),
                ("deeponet", "siren", "causal"),
                ("deepokan", None, "seq"),
                ("deepokan", None, "causal"),
            ],
            "variant_a_spec": ("soap", "baseline"),
            "variant_b_spec": ("soap", "baseline"),
            "special_mode": "causal",
        },
    ]


def get_row_label(arch: str, activation: Optional[str] = None) -> str:
    """Get display label for a row."""
    label = ARCH_LABELS.get(arch, arch)
    if arch == "deeponet" and activation:
        from src.core.visualization import ACTIVATION_LABELS

        label = f"{label} ({ACTIVATION_LABELS.get(activation, activation)})"
    return label


def get_row_color(arch: str) -> str:
    """Get color for an architecture row."""
    return ARCH_COLORS.get(arch, "gray")


def load_comparison_predictions(
    comparison: Dict,
    seed: int,
    device: str,
    test_loaders: Dict,
    targets_dict: Dict,
) -> Tuple[List[Dict], Optional[np.ndarray]]:
    """Load variant_a and variant_b predictions for all rows in a comparison.

    Returns:
        row_data: list of dicts with 'label', 'color', 'preds_a', 'preds_b'
        targets: shared target array [N_test, T] (None if no models loaded)
    """
    checkpoint_dir = project_root / "checkpoints" / f"soap_vs_adam_ablation_seed{seed}"
    opt_a, loss_a = comparison["variant_a_spec"]
    opt_b, loss_b = comparison["variant_b_spec"]
    special = comparison.get("special_mode")

    row_data = []
    targets = None

    if special == "activation":
        # Rows = [Baseline, BSP], columns = [Tanh, SIREN]
        for loss_name in ["baseline", "bsp"]:
            loss_label = "Baseline" if loss_name == "baseline" else "BSP"
            # Tanh = variant_a, SIREN = variant_b
            model_a = load_trained_model(
                "deeponet",
                "soap",
                loss_name,
                checkpoint_dir,
                device=device,
                activation="tanh",
                causal_mode="causal",
            )
            model_b = load_trained_model(
                "deeponet",
                "soap",
                loss_name,
                checkpoint_dir,
                device=device,
                activation="siren",
                causal_mode="causal",
            )

            loader = test_loaders["deeponet"]
            if targets is None:
                targets = targets_dict["deeponet"]

            rd = {
                "label": loss_label,
                "color": ARCH_COLORS["deeponet"],
                "preds_a": None,
                "preds_b": None,
            }

            if model_a is not None:
                preds_a, _ = get_predictions(
                    model_a, loader, device, use_causal=True, arch="deeponet"
                )
                rd["preds_a"] = preds_a
            if model_b is not None:
                preds_b, _ = get_predictions(
                    model_b, loader, device, use_causal=True, arch="deeponet"
                )
                rd["preds_b"] = preds_b

            row_data.append(rd)

    elif special == "causal":
        # Rows = [DeepONet, DeepOKAN], variant_a = seq, variant_b = causal
        for arch_base, activation in [("deeponet", "siren"), ("deepokan", None)]:
            loader = test_loaders[arch_base]
            if targets is None:
                targets = targets_dict[arch_base]

            rd = {
                "label": get_row_label(arch_base, activation),
                "color": get_row_color(arch_base),
                "preds_a": None,
                "preds_b": None,
            }

            # Variant A: non-causal (sequence mode)
            model_a = load_trained_model(
                arch_base,
                opt_a,
                loss_a,
                checkpoint_dir,
                device=device,
                activation=activation,
                causal_mode="seq",
            )
            if model_a is not None:
                preds_a, _ = get_predictions(
                    model_a, loader, device, use_causal=False, arch=arch_base
                )
                rd["preds_a"] = preds_a

            # Variant B: causal (per-timestep mode)
            model_b = load_trained_model(
                arch_base,
                opt_b,
                loss_b,
                checkpoint_dir,
                device=device,
                activation=activation,
                causal_mode="causal",
            )
            if model_b is not None:
                preds_b, _ = get_predictions(
                    model_b, loader, device, use_causal=True, arch=arch_base
                )
                rd["preds_b"] = preds_b

            row_data.append(rd)

    else:
        # Standard mode: each arch is a row, variant_a vs variant_b differ by
        # optimizer or loss
        for arch, activation, causal_mode in comparison["archs"]:
            is_deeponet_like = arch in ("deeponet", "deepokan")
            use_causal = (causal_mode == "causal") if is_deeponet_like else False
            loader = test_loaders[arch]
            if targets is None:
                targets = targets_dict[arch]

            rd = {
                "label": get_row_label(
                    arch, activation if arch == "deeponet" else None
                ),
                "color": get_row_color(arch),
                "preds_a": None,
                "preds_b": None,
            }

            model_a = load_trained_model(
                arch,
                opt_a,
                loss_a,
                checkpoint_dir,
                device=device,
                activation=activation,
                causal_mode=causal_mode,
            )
            if model_a is not None:
                preds_a, _ = get_predictions(
                    model_a, loader, device, use_causal=use_causal, arch=arch
                )
                rd["preds_a"] = preds_a

            model_b = load_trained_model(
                arch,
                opt_b,
                loss_b,
                checkpoint_dir,
                device=device,
                activation=activation,
                causal_mode=causal_mode,
            )
            if model_b is not None:
                preds_b, _ = get_predictions(
                    model_b, loader, device, use_causal=use_causal, arch=arch
                )
                rd["preds_b"] = preds_b

            row_data.append(rd)

    return row_data, targets


def generate_comparison_figure(
    comparison: Dict,
    row_data: List[Dict],
    targets: np.ndarray,
    sample_idx: int,
    output_path: str,
) -> None:
    """Generate a single comparison figure using the plotting utility."""
    rows = []
    for rd in row_data:
        rows.append(
            {
                "label": rd["label"],
                "color": rd["color"],
                "variant_a": rd.get("preds_a"),
                "variant_b": rd.get("preds_b"),
            }
        )

    plot_fixed_sample_comparison_grid(
        rows=rows,
        targets=targets,
        sample_idx=sample_idx,
        output_path=output_path,
        group_a_label=comparison["group_a_label"],
        group_b_label=comparison["group_b_label"],
        title=comparison["title"],
        dt=0.02,
    )
    print(f"  Saved: {output_path}")


COMPACT_MODEL_SPECS = [
    ("don_siren", "deeponet", "soap", "baseline", "siren", "causal"),
    ("dokan_soap", "deepokan", "soap", "baseline", None, "causal"),
    ("fno_base", "fno", "soap", "baseline", None, None),
    ("cno_base", "cno", "soap", "baseline", None, None),
    ("fno_bsp", "fno", "soap", "bsp", None, None),
    ("don_tanh", "deeponet", "soap", "baseline", "tanh", "causal"),
    ("dokan_adam", "deepokan", "adam", "baseline", None, "causal"),
]

COMPACT_ROWS = [
    {
        "label": "Impact of\nArchitecture",
        "models": [
            # (model_key, display_name, color)
            ("don_siren", "DeepONet", ARCH_COLORS["deeponet"]),
            ("dokan_soap", "DeepOKAN", ARCH_COLORS["deepokan"]),
            ("fno_base", "FNO", ARCH_COLORS["fno"]),
            ("cno_base", "CNO", ARCH_COLORS["cno"]),
        ],
    },
    {
        "label": "Impact of\nBSP",
        "models": [
            ("fno_base", "Baseline", LOSS_COLORS_DEFAULT["baseline"]),
            ("fno_bsp", "BSP", LOSS_COLORS_DEFAULT["bsp"]),
        ],
    },
    {
        "label": "Impact of\nOptimizer",
        "models": [
            ("dokan_adam", "Adam", OPTIMIZER_COLORS["adam"]),
            ("dokan_soap", "SOAP", OPTIMIZER_COLORS["soap"]),
        ],
    },
    {
        "label": "Impact of\nActivation",
        "models": [
            ("don_tanh", "Tanh", ACTIVATION_COLORS["tanh"]),
            ("don_siren", "SIREN", ACTIVATION_COLORS["siren"]),
        ],
    },
]


def generate_compact_impact_figure(
    device: str,
    test_loaders: Dict,
    targets_dict: Dict,
    sample: int,
    seed: int,
    output_dir: Path,
) -> None:
    """Generate the compact 2x4 impact figure (all rows share the fixed sample)."""
    print(f"\n{'='*70}")
    print("Generating compact impact figure")
    print(f"{'='*70}")

    checkpoint_dir = project_root / "checkpoints" / f"soap_vs_adam_ablation_seed{seed}"
    preds = {}
    for key, arch, opt, loss, activation, causal_mode in COMPACT_MODEL_SPECS:
        model = load_trained_model(
            arch,
            opt,
            loss,
            checkpoint_dir,
            device=device,
            activation=activation,
            causal_mode=causal_mode,
        )
        if model is None:
            preds[key] = None
            continue
        use_causal = arch in ("deeponet", "deepokan") and causal_mode == "causal"
        p, _ = get_predictions(
            model, test_loaders[arch], device, use_causal=use_causal, arch=arch
        )
        preds[key] = p

    target_1d = _extract_sample(targets_dict["deeponet"], sample)

    plot_rows = []
    plot_targets = []
    for row_spec in COMPACT_ROWS:
        variants = []
        for key, display_name, color in row_spec["models"]:
            if preds.get(key) is None:
                print(f"    Warning: {key} not available, skipping")
                continue
            pred_1d = _extract_sample(preds[key], sample)
            variants.append((display_name, color, "-", pred_1d))

        if not variants:
            print(f"    Warning: no variants for '{row_spec['label']}', skipping")
            continue

        plot_rows.append({"label": row_spec["label"], "variants": variants})
        plot_targets.append(target_1d)

    if not plot_rows:
        print("  Warning: no valid rows, skipping compact figure")
        return

    output_path = str(output_dir / "compact_impact.png")
    plot_compact_impact_grid(
        rows=plot_rows, targets=plot_targets, output_path=output_path, dt=0.02
    )
    print(f"\n  Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Create illustrative comparison figures for paper claims"
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=DISPLAY_SAMPLE,
        help=f"Test sample index to display (default: {DISPLAY_SAMPLE})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DISPLAY_SEED,
        help=f"Seed whose checkpoints to load (default: {DISPLAY_SEED})",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(project_root / "results" / "illustrative_figures"),
        help="Output directory for figures",
    )
    args = parser.parse_args()

    sample = args.sample
    seed = args.seed
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Sample: {sample}, Seed: {seed}")
    print(f"Output: {output_dir}")

    print("\nLoading test data...")
    data_dir = project_root / "CDONData"

    # Standard architectures need standard loader; CNO needs 4096 loader
    test_loaders = {}
    targets_dict = {}

    for arch_key, target_len in [("standard", None), ("cno", CNO_INTERNAL_SIZE)]:
        dataset = CDONDataset(
            data_dir=str(data_dir),
            split="test",
            mode="sequence",
            signal_length=SIGNAL_LENGTH_CDON,
            target_signal_length=target_len,
        )
        loader = create_test_dataloader(dataset, batch_size=BATCH_SIZE_SEQUENCE)

        # Extract targets from the loader
        with torch.no_grad():
            all_targets = []
            for batch in loader:
                _, tgts, _ = batch
                all_targets.append(tgts)
            raw_targets = torch.cat(all_targets, dim=0)

        if arch_key == "standard":
            std_targets = raw_targets.numpy()
            # Shared by deeponet, deepokan, fno
            test_loaders["deeponet"] = loader
            test_loaders["deepokan"] = loader
            test_loaders["fno"] = loader
            targets_dict["deeponet"] = std_targets
            targets_dict["deepokan"] = std_targets
            targets_dict["fno"] = std_targets
        else:
            test_loaders["cno"] = loader
            # CNO targets are at 4096 in the loader; post-process to 4000
            cno_targets = postprocess_cno_predictions(raw_targets).numpy()
            targets_dict["cno"] = cno_targets

    print(
        f"  Standard test set: {std_targets.shape[0]} samples, {std_targets.shape[-1]} timesteps"
    )
    print(
        f"  CNO test set: {targets_dict['cno'].shape[0]} samples, {targets_dict['cno'].shape[-1]} timesteps"
    )

    comparisons = define_comparisons()

    for comp in comparisons:
        name = comp["name"]
        print(f"\n{'='*70}")
        print(f"Comparison: {name}")
        print(f"  {comp['group_a_label']} vs {comp['group_b_label']}")
        print(f"{'='*70}")

        row_data, targets = load_comparison_predictions(
            comp, seed, device, test_loaders, targets_dict
        )
        if targets is None:
            print(f"  Warning: no models loaded for {name}, skipping")
            continue
        output_path = str(output_dir / f"{name}_sample{sample}_seed{seed}.png")
        generate_comparison_figure(comp, row_data, targets, sample, output_path)

    generate_compact_impact_figure(
        device, test_loaders, targets_dict, sample, seed, output_dir
    )

    print(f"\n{'='*70}")
    print(f"All figures saved to: {output_dir}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
