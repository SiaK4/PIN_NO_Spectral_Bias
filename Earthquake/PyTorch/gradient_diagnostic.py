#!/usr/bin/env python3
"""
Gradient cosine similarity diagnostic.

Measures cosine similarity between L2 field loss and BSP loss gradients during
training, to test whether per-timestep field loss and sequence-level spectral
loss produce competing gradient signals for causal architectures.

Usage:
    python gradient_diagnostic.py --index 0 --device cuda
"""

import argparse
import copy
import json
import sys
import time
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F

project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

from configs.loss_config import LOSS_CONFIG_MAP, LossConfig
from configs.model_configs import MODEL_CONFIGS
from src.core.evaluation.constants import (
    BATCH_SIZE_SEQUENCE,
    CNO_INTERNAL_SIZE,
    LX_CDON,
    SIGNAL_LENGTH_CDON,
)
from src.core.evaluation.loss_factory import compute_bsp_loss
from src.core.models.model_factory import create_model
from src.core.training.experiment_utils import (
    create_dataloaders,
    create_trainer,
    get_run_name,
    prepare_model,
    print_header,
)

CONFIGS = [
    ("deeponet", "siren", "causal", 500),
    ("deeponet", "tanh", "causal", 500),
    ("deepokan", None, "causal", 500),
    ("fno", None, None, 1500),
    ("cno", None, None, 1500),
    ("deeponet", "siren", "noncausal", 1500),
    ("deeponet", "tanh", "noncausal", 1500),
    ("deepokan", None, "noncausal", 1500),
]


def compute_gradient_cosine_similarity(
    model: torch.nn.Module,
    l2_loss: torch.Tensor,
    bsp_loss: torch.Tensor,
) -> Tuple[float, float, float]:
    """Compute cosine similarity between L2 and BSP gradients.

    Uses torch.autograd.grad to get per-loss gradients without modifying
    the model's .grad attributes.

    Returns:
        cosine_sim: scalar cosine similarity between flattened gradient vectors
        l2_grad_norm: L2 norm of field loss gradient
        bsp_grad_norm: L2 norm of BSP loss gradient
    """
    params = [p for p in model.parameters() if p.requires_grad]

    l2_grads = torch.autograd.grad(
        l2_loss, params, retain_graph=True, allow_unused=True
    )
    bsp_grads = torch.autograd.grad(
        bsp_loss, params, retain_graph=True, allow_unused=True
    )

    l2_flat = []
    bsp_flat = []
    for lg, bg, p in zip(l2_grads, bsp_grads, params):
        l2_flat.append(
            lg.flatten() if lg is not None else torch.zeros_like(p).flatten()
        )
        bsp_flat.append(
            bg.flatten() if bg is not None else torch.zeros_like(p).flatten()
        )

    l2_vec = torch.cat(l2_flat)
    bsp_vec = torch.cat(bsp_flat)

    l2_norm = l2_vec.norm()
    bsp_norm = bsp_vec.norm()

    if l2_norm < 1e-12 or bsp_norm < 1e-12:
        return 0.0, float(l2_norm), float(bsp_norm)

    cosine_sim = F.cosine_similarity(l2_vec.unsqueeze(0), bsp_vec.unsqueeze(0)).item()
    return cosine_sim, float(l2_norm), float(bsp_norm)


def _filter_zero_pairs(seq_inputs, seq_targets, sample_indices):
    """Filter out zero signal pairs (sample index 0) from batch.

    Matches SimpleTrainer._filter_zero_pairs().
    """
    non_zero_mask = sample_indices != 0
    if non_zero_mask.sum() == 0:
        return None
    return (
        seq_inputs[non_zero_mask],
        seq_targets[non_zero_mask],
        sample_indices[non_zero_mask],
    )


def run_diagnostic(index, device="cuda", seed=1, log_every=1):
    """Run gradient cosine similarity diagnostic for a single config."""
    arch, activation, causal_mode, epochs = CONFIGS[index]

    run_name = get_run_name(
        arch, "soap", "bsp", activation, causal_mode, suffix="grad_diag"
    )
    is_deeponet_like = arch in ["deeponet", "deepokan"]
    use_causal = (causal_mode == "causal") if causal_mode else False
    is_cno = arch == "cno"

    config_str = "SOAP + BSP"
    if is_deeponet_like:
        config_str = f"{'causal' if use_causal else 'sequence'} + {config_str}"
    if arch == "deeponet" and activation:
        config_str = f"{activation} + {config_str}"
    print_header("GRADIENT COSINE SIMILARITY", arch, config_str)
    print(f"Epochs: {epochs}, Device: {device}, Seed: {seed}")

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
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Loss config (BSP)
    if is_cno:
        signal_length = CNO_INTERNAL_SIZE
    else:
        signal_length = SIGNAL_LENGTH_CDON

    base_loss_config = LOSS_CONFIG_MAP["bsp"]
    loss_config = LossConfig(
        loss_type=base_loss_config.loss_type,
        loss_params=copy.deepcopy(base_loss_config.loss_params),
        description=base_loss_config.description,
    )
    if is_cno:
        loss_config.loss_params["signal_length"] = signal_length

    print("\n2. Loading physical-space data...")
    loaders = create_dataloaders(
        arch,
        batch_size=BATCH_SIZE_SEQUENCE,
        is_baseline=False,
        use_causal=use_causal,
    )

    trainer = create_trainer(
        model=model,
        loaders=loaders,
        loss_config=loss_config,
        experiment_name=run_name,
        checkpoint_dir=project_root / "checkpoints" / "gradient_diagnostic",
        epochs=epochs,
        device=device,
        arch=arch,
        optimizer_override="soap",
        use_causal=use_causal,
        max_training_time=540,
        seed=seed,
    )

    max_training_time = 540
    print(
        f"\n3. Training with gradient diagnostic (logging every {log_every} batches, "
        f"max {max_training_time}s)..."
    )
    results = []
    model.train()
    training_start = time.time()

    for epoch in range(epochs):
        elapsed = time.time() - training_start
        if elapsed >= max_training_time:
            print(
                f"\n  Time limit reached ({elapsed:.0f}s >= {max_training_time}s) "
                f"at epoch {epoch}/{epochs}"
            )
            break
        epoch_cosines = []
        epoch_l2_norms = []
        epoch_bsp_norms = []
        epoch_l2_losses = []
        epoch_bsp_losses = []

        if is_deeponet_like and use_causal:
            seq_loader = loaders["sequence_train"]
            per_ts_dataset = loaders["per_timestep_train_dataset"]

            for batch_idx, sequence_batch in enumerate(seq_loader):
                # Format: input, target, index
                seq_inputs = sequence_batch[0].to(device)  # [B, 1, 4000]
                seq_targets = sequence_batch[1].to(device)  # [B, 1, 4000]
                sample_indices = sequence_batch[2].to(device)  # [B]

                filtered = _filter_zero_pairs(seq_inputs, seq_targets, sample_indices)
                if filtered is None:
                    continue
                seq_inputs, seq_targets, sample_indices = filtered

                # Convert sequence indices to earthquake indices
                earthquake_indices = sample_indices - 1

                per_ts_batch = per_ts_dataset.get_all_timesteps_for_earthquakes(
                    earthquake_indices=earthquake_indices
                )
                per_ts_inputs = per_ts_batch["input"].to(device)  # [B*T, 4000]
                per_ts_targets = per_ts_batch["target"].to(device)  # [B*T]
                per_ts_time_coords = per_ts_batch["time_coord"].to(device)  # [B*T]

                # Per-timestep forward (L2 field loss)
                B = seq_inputs.shape[0]
                T_per_eq = per_ts_inputs.shape[0] // B

                trainer.optimizer.zero_grad()

                if hasattr(model, "forward_per_timestep_batched"):
                    unique_time_coords = per_ts_time_coords[:T_per_eq]
                    per_ts_outputs = model.forward_per_timestep_batched(
                        per_ts_inputs,
                        n_earthquakes=B,
                        timesteps_per_earthquake=T_per_eq,
                        time_coords=unique_time_coords,
                    )
                else:
                    per_ts_outputs = model.forward_per_timestep(
                        per_ts_inputs, per_ts_time_coords
                    )
                per_ts_outputs = per_ts_outputs.squeeze(-1)  # [B*T]

                # L2 field loss (global L2 norm).
                diff = (per_ts_targets - per_ts_outputs).view(B, T_per_eq)
                l2_loss = torch.norm(diff, p=2)

                # Sequence BSP loss (causal sequence forward)
                seq_outputs = model.forward_causal_sequence(seq_inputs)
                bsp_loss = compute_bsp_loss(
                    seq_outputs, seq_targets, lx=LX_CDON, reduction="sum"
                )

                if batch_idx % log_every == 0:
                    cos_sim, l2_norm, bsp_norm = compute_gradient_cosine_similarity(
                        model, l2_loss, bsp_loss
                    )
                    epoch_cosines.append(cos_sim)
                    epoch_l2_norms.append(l2_norm)
                    epoch_bsp_norms.append(bsp_norm)
                    epoch_l2_losses.append(float(l2_loss.item()))
                    epoch_bsp_losses.append(float(bsp_loss.item()))

                combined = l2_loss + trainer.w_bsp * bsp_loss
                combined.backward()
                trainer.optimizer.step()

        else:
            train_loader = loaders["train"]

            for batch_idx, batch in enumerate(train_loader):
                # Format: input, target, index
                seq_inputs = batch[0].to(device)
                seq_targets = batch[1].to(device)
                sample_indices = batch[2].to(device) if len(batch) == 3 else None

                if sample_indices is not None:
                    filtered = _filter_zero_pairs(
                        seq_inputs, seq_targets, sample_indices
                    )
                    if filtered is None:
                        continue
                    seq_inputs, seq_targets, sample_indices = filtered

                trainer.optimizer.zero_grad()

                seq_outputs = model(seq_inputs)  # [B, 1, T]

                diff = seq_targets - seq_outputs
                l2_loss = torch.norm(diff, p=2)

                bsp_loss = compute_bsp_loss(seq_outputs, seq_targets, lx=LX_CDON)

                if batch_idx % log_every == 0:
                    cos_sim, l2_norm, bsp_norm = compute_gradient_cosine_similarity(
                        model, l2_loss, bsp_loss
                    )
                    epoch_cosines.append(cos_sim)
                    epoch_l2_norms.append(l2_norm)
                    epoch_bsp_norms.append(bsp_norm)
                    epoch_l2_losses.append(float(l2_loss.item()))
                    epoch_bsp_losses.append(float(bsp_loss.item()))

                combined = l2_loss + trainer.w_bsp * bsp_loss
                combined.backward()
                trainer.optimizer.step()

        if trainer.scheduler is not None:
            trainer.scheduler.step()

        if epoch_cosines:
            mean_cos = np.mean(epoch_cosines)
            std_cos = np.std(epoch_cosines)
            mean_l2_norm = np.mean(epoch_l2_norms)
            mean_bsp_norm = np.mean(epoch_bsp_norms)
            mean_l2_loss = np.mean(epoch_l2_losses)
            mean_bsp_loss = np.mean(epoch_bsp_losses)

            results.append(
                {
                    "epoch": epoch,
                    "cosine_sim_mean": float(mean_cos),
                    "cosine_sim_std": float(std_cos),
                    "cosine_sim_values": [float(c) for c in epoch_cosines],
                    "l2_grad_norm_mean": float(mean_l2_norm),
                    "bsp_grad_norm_mean": float(mean_bsp_norm),
                    "l2_loss_mean": float(mean_l2_loss),
                    "bsp_loss_mean": float(mean_bsp_loss),
                }
            )

            if epoch % 10 == 0 or epoch == epochs - 1:
                print(
                    f"  Epoch {epoch:4d}: cos_sim={mean_cos:+.4f} (+/-{std_cos:.4f}), "
                    f"L2_grad={mean_l2_norm:.2e}, BSP_grad={mean_bsp_norm:.2e}, "
                    f"L2_loss={mean_l2_loss:.4e}, BSP_loss={mean_bsp_loss:.4e}"
                )

    output_dir = project_root / "results" / "gradient_diagnostic"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{run_name}.json"

    output = {
        "config": {
            "arch": arch,
            "activation": activation,
            "causal_mode": causal_mode,
            "optimizer": "soap",
            "loss": "bsp",
            "epochs": epochs,
            "seed": seed,
        },
        "history": results,
        "summary": {
            "mean_cosine_sim": float(np.mean([r["cosine_sim_mean"] for r in results])),
            "std_cosine_sim": float(np.std([r["cosine_sim_mean"] for r in results])),
            "final_cosine_sim": results[-1]["cosine_sim_mean"] if results else None,
        },
    }

    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {output_file}")
    print(
        f"Overall mean cosine similarity: {output['summary']['mean_cosine_sim']:+.4f}"
    )

    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Gradient cosine similarity diagnostic"
    )
    parser.add_argument("--index", type=int, required=True, help="Config index (0-4)")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--log-every",
        type=int,
        default=1,
        help="Log cosine similarity every N batches (default: 1)",
    )
    args = parser.parse_args()

    if args.index < 0 or args.index >= len(CONFIGS):
        print(f"Error: index must be 0-{len(CONFIGS)-1}")
        sys.exit(1)

    run_diagnostic(
        index=args.index,
        device=args.device,
        seed=args.seed,
        log_every=args.log_every,
    )
