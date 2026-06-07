#!/usr/bin/env python3
"""
Plot gradient cosine similarity vs training completion %.

Loads all gradient diagnostic JSONs from results/gradient_diagnostic/,
normalizes epoch to completion % (epoch / total_epochs * 100), and
creates a single overlaid plot with all configs.

Usage:
    python plot_gradient.py
    python plot_gradient.py --results-dir results/gradient_diagnostic
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

from src.core.visualization import ARCH_COLORS

# Target number of points per curve after smoothing
TARGET_POINTS = 40

# filename, display_label, arch_key, is_causal
PLOT_CONFIGS = [
    (
        "deeponet_causal_soap_bsp_grad_diag.json",
        "DeepONet (SIREN) causal",
        "deeponet",
        True,
    ),
    (
        "deeponet_tanh_causal_soap_bsp_grad_diag.json",
        "DeepONet (Tanh) causal",
        "deeponet",
        True,
    ),
    ("deepokan_causal_soap_bsp_grad_diag.json", "DeepOKAN causal", "deepokan", True),
    ("fno_soap_bsp_grad_diag.json", "FNO", "fno", False),
    ("cno_soap_bsp_grad_diag.json", "CNO", "cno", False),
    (
        "deeponet_noncausal_soap_bsp_grad_diag.json",
        "DeepONet (SIREN) non-causal",
        "deeponet",
        False,
    ),
    (
        "deeponet_tanh_noncausal_soap_bsp_grad_diag.json",
        "DeepONet (Tanh) non-causal",
        "deeponet",
        False,
    ),
    (
        "deepokan_noncausal_soap_bsp_grad_diag.json",
        "DeepOKAN non-causal",
        "deepokan",
        False,
    ),
]

LABEL_COLORS = {
    "DeepONet (SIREN) causal": "#1f77b4",
    "DeepONet (Tanh) causal": "#6baed6",
    "DeepOKAN causal": "#ff7f0e",
    "FNO": "#2ca02c",
    "CNO": "#d62728",
    "DeepONet (SIREN) non-causal": "#1f77b4",
    "DeepONet (Tanh) non-causal": "#6baed6",
    "DeepOKAN non-causal": "#ff7f0e",
}


def main():
    parser = argparse.ArgumentParser(
        description="Plot gradient cosine similarity diagnostics"
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=project_root / "results" / "gradient_diagnostic",
        help="Directory containing gradient diagnostic JSON files",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output PNG path",
    )
    args = parser.parse_args()

    results_dir = args.results_dir
    output_path = args.output or results_dir / "cosine_sim_vs_completion.png"

    _fig, ax = plt.subplots(figsize=(10, 6))

    loaded_any = False
    for filename, label, arch_key, is_causal in PLOT_CONFIGS:
        filepath = results_dir / filename
        if not filepath.exists():
            print(f"  Skipping (not found): {filename}")
            continue

        with open(filepath) as f:
            data = json.load(f)

        history = data["history"]
        if not history:
            print(f"  Skipping (empty history): {filename}")
            continue

        cosine_sims = np.array([h["cosine_sim_mean"] for h in history])
        n_epochs = len(cosine_sims)

        # Auto bin size to land near TARGET_POINTS
        bin_size = max(1, n_epochs // TARGET_POINTS)
        n_bins = n_epochs // bin_size

        if n_bins > 1:
            trimmed = n_bins * bin_size
            cosine_sims = cosine_sims[:trimmed].reshape(n_bins, bin_size).mean(axis=1)
        else:
            n_bins = 1
            cosine_sims = np.array([cosine_sims.mean()])

        completion_pct = np.linspace(0, 100, n_bins)

        color = LABEL_COLORS.get(label, ARCH_COLORS.get(arch_key, "gray"))
        linestyle = "--" if is_causal else "-"

        ax.plot(
            completion_pct,
            cosine_sims,
            color=color,
            linestyle=linestyle,
            linewidth=1.5,
            label=label,
            alpha=0.85,
        )
        loaded_any = True
        print(
            f"  Loaded: {filename} ({n_epochs} epochs, "
            f"bin_size={bin_size}, {n_bins} points)"
        )

    if not loaded_any:
        print("No gradient diagnostic results found.")
        return

    ax.axhline(y=0, color="gray", linestyle=":", linewidth=1.0, alpha=0.7)

    ax.set_xlabel("Training Completion (%)", fontsize=13)
    ax.set_ylabel("Cosine Similarity", fontsize=13)
    ax.set_title(
        "Gradient Cosine Similarity: L2 vs BSP", fontsize=15, fontweight="bold"
    )
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        fontsize=9,
        framealpha=0.9,
        ncol=3,
    )
    ax.set_xlim(0, 100)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=11)

    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()
