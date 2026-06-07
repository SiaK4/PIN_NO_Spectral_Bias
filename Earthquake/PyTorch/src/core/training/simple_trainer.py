"""
Simple trainer for the models.

- DeepONet: Per-timestep (field) + Full-sequence (BSP) batches
- FNO/CNO: Full-sequence batches (both field and BSP computed on sequences)
"""

import time
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple, cast

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

if TYPE_CHECKING:
    from ..data_processing.cdon_dataset import CDONDataset

from pathlib import Path

from configs.loss_config import LossConfig
from configs.training_config import TrainingConfig
from rich.console import Console
from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn

from ..evaluation.constants import EPSILON, LX_CDON
from ..evaluation.loss_factory import compute_bsp_loss, create_loss
from ..evaluation.metrics import compute_field_nrmse
from ..models.model_factory import model_supports_per_timestep
from ..utils.reproducibility import set_global_seed
from .optimizers.optimizer_factory import create_optimizer


class SimpleTrainer:
    """Trainer for per-timestep DeepO and sequence-only neural operators."""

    def __init__(
        self,
        model: nn.Module,
        per_timestep_train_loader: Optional[DataLoader] = None,
        sequence_train_loader: Optional[DataLoader] = None,
        config: Optional[TrainingConfig] = None,
        loss_config: Optional[LossConfig] = None,
        experiment_name: str = "experiment",
        train_loader: Optional[DataLoader] = None,
        test_loader: Optional[DataLoader] = None,
        use_causal: bool = True,
    ):
        """Initialize trainer state, optimizer, scheduler, and checkpoint paths."""
        self.use_causal = use_causal

        assert config is not None, "config (TrainingConfig) is required"
        assert loss_config is not None, "loss_config (LossConfig) is required"

        set_global_seed(
            seed=config.seed, deterministic=config.deterministic, verbose=config.verbose
        )

        if train_loader is not None and sequence_train_loader is None:
            sequence_train_loader = train_loader

        has_sequence_train = sequence_train_loader is not None
        has_per_timestep_train = per_timestep_train_loader is not None

        if not has_sequence_train and not has_per_timestep_train:
            raise ValueError(
                "Must provide at least one training loader: "
                "'sequence_train_loader'/'train_loader' OR 'per_timestep_train_loader'"
            )

        self.has_test_selection = test_loader is not None

        self.model = model
        self.per_timestep_train_loader = per_timestep_train_loader
        self.sequence_train_loader = sequence_train_loader
        self.test_loader = test_loader
        self.config = config
        self.experiment_name = experiment_name

        self.per_timestep_train_dataset = (
            per_timestep_train_loader.dataset
            if per_timestep_train_loader is not None
            else None
        )

        self.is_deeponet = model_supports_per_timestep(model) and self.use_causal
        self.is_sequence_only = not self.is_deeponet

        if self.is_sequence_only:
            assert (
                sequence_train_loader is not None
            ), "sequence_train_loader required for sequence-only models (FNO, CNO, or DeepONet with use_causal=False)"

        if config.device == "cuda" and not torch.cuda.is_available():
            print("Warning: CUDA requested but not available. Falling back to CPU.")
            self.device = torch.device("cpu")
        else:
            self.device = torch.device(config.device)

        self.model.to(self.device)

        self.criterion = create_loss(loss_config)
        self.criterion.to(self.device)

        self.loss_config = loss_config

        self.w_field = loss_config.loss_params.get("w_field", 1.0)
        self.w_bsp = loss_config.loss_params.get("w_bsp", 1.0)

        self.optimizer = create_optimizer(
            optimizer_type=config.optimizer_type,
            model_parameters=self.model.parameters(),
            config=config,
        )

        use_dual_batch_init = (
            self.is_deeponet
            and self.per_timestep_train_loader is not None
            and loss_config is not None
            and loss_config.loss_type == "bsp"
        )

        if use_dual_batch_init:
            assert (
                self.sequence_train_loader is not None
            ), "sequence_train_loader required for dual-batch"
            steps_per_epoch = len(self.sequence_train_loader)
        elif self.is_deeponet and self.per_timestep_train_loader is not None:
            steps_per_epoch = len(self.per_timestep_train_loader)
        else:
            assert (
                self.sequence_train_loader is not None
            ), "sequence_train_loader required for sequence-only models"
            steps_per_epoch = len(self.sequence_train_loader)

        self.scheduler = self._create_scheduler(steps_per_epoch)

        self.checkpoint_dir = Path(config.checkpoint_dir) / experiment_name
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.current_epoch = 0
        self.best_test_nrmse = float("inf")
        self.train_history = []
        self.test_history = []

        self.console = Console()

    def _check_outputs_for_instability(
        self, outputs: torch.Tensor, inputs: torch.Tensor, epoch: int, batch_idx: int
    ) -> bool:
        """
        Check outputs for Inf/NaN before loss computation (early detection).

        Args:
            outputs: Model outputs to check
            inputs: Model inputs for diagnostic info
            epoch: Current epoch number
            batch_idx: Current batch index

        Returns:
            True if instability detected, False otherwise
        """
        if torch.isinf(outputs).any():
            self.console.print(
                "\n[bold red]❌ Inf detected in model outputs![/bold red]"
            )
            self.console.print(f"  Epoch: {epoch}, Batch: {batch_idx}")
            self.console.print(
                f"  Output range: [{outputs.min():.6e}, {outputs.max():.6e}]"
            )
            self.console.print(
                f"  Input range: [{inputs.min():.6e}, {inputs.max():.6e}]"
            )

            corrupt_params = []
            for name, param in self.model.named_parameters():
                if torch.isnan(param).any() or torch.isinf(param).any():
                    corrupt_params.append(name)

            if corrupt_params:
                self.console.print(
                    f"  [red]Corrupted parameters:[/red] {corrupt_params[:5]}"
                )

            self.console.print("\n[yellow]Likely causes:[/yellow]")
            self.console.print(
                f"  1. Learning rate too high (current: {self.optimizer.param_groups[0]['lr']:.2e})"
            )
            self.console.print("  2. Model weights exploded from previous batches")
            self.console.print(
                "  3. Activation function instability (try 'tanh' instead of 'ReLU')"
            )

            return True

        if torch.isnan(outputs).any():
            self.console.print(
                "\n[bold red]❌ NaN detected in model outputs![/bold red]"
            )
            self.console.print(f"  Epoch: {epoch}, Batch: {batch_idx}")
            self.console.print(
                "  This usually means model parameters were already corrupted"
            )
            return True

        return False

    def _check_model_parameters(self, epoch: int, batch_idx: int) -> bool:
        """
        Check model parameters for NaN/Inf corruption.

        Args:
            epoch: Current epoch number
            batch_idx: Current batch index

        Returns:
            True if corruption detected, False otherwise
        """
        corrupt_params = []
        for name, param in self.model.named_parameters():
            if torch.isnan(param).any() or torch.isinf(param).any():
                corrupt_params.append(name)

        if corrupt_params:
            self.console.print("\n[bold red]❌ Model parameters corrupted![/bold red]")
            self.console.print(f"  Epoch: {epoch}, Batch: {batch_idx}")
            self.console.print(f"  Corrupted parameters: {corrupt_params[:10]}")
            self.console.print(
                "\n[yellow]Recovery not possible - model weights are corrupted[/yellow]"
            )
            self.console.print(
                "  Reduce learning rate and restart training from checkpoint"
            )
            return True

        return False

    def _check_for_nan(
        self, loss: torch.Tensor, loss_name: str, epoch: int, batch_idx: int
    ) -> bool:
        """
        Check if loss contains NaN and print diagnostic information.

        Args:
            loss: Loss tensor to check
            loss_name: Name of the loss for reporting (e.g., 'field_loss', 'bsp_loss')
            epoch: Current epoch number
            batch_idx: Current batch index

        Returns:
            True if NaN detected, False otherwise
        """
        if torch.isnan(loss).any() or torch.isinf(loss).any():
            self.console.print(
                f"\n[bold red]❌ NaN/Inf detected in {loss_name}![/bold red]"
            )
            self.console.print(f"  Epoch: {epoch}, Batch: {batch_idx}")
            self.console.print(
                f"  Loss value: {loss.item() if loss.numel() == 1 else loss}"
            )
            self.console.print(
                f"  Learning rate: {self.optimizer.param_groups[0]['lr']:.2e}"
            )

            nan_params = []
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                        nan_params.append(name)

            if nan_params:
                self.console.print(
                    f"  Parameters with NaN/Inf gradients: {nan_params[:5]}..."
                )

            total_norm = 0.0
            for p in self.model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm**0.5
            self.console.print(f"  Total gradient norm: {total_norm:.2e}")

            self.console.print("\n[yellow]Diagnostic Tips:[/yellow]")
            self.console.print(
                f"  1. Reduce learning rate (current: {self.optimizer.param_groups[0]['lr']:.2e})"
            )
            self.console.print("     Suggested: Try 1e-4 or 3e-4 for SOAP optimizer")
            self.console.print(
                f"  2. Gradient clipping: {'ENABLED' if self.config.max_grad_norm > 0 else 'DISABLED'} (max_norm={self.config.max_grad_norm})"
            )
            self.console.print("  3. Check input data for NaN/Inf values")
            self.console.print("  4. Increase epsilon in loss config if using BSP")
            self.console.print(
                "  5. Try Adam optimizer instead of SOAP (optimizer_type='adam' in config)"
            )

            return True
        return False

    def _create_scheduler(self, steps_per_epoch: int):
        """
        Create learning rate scheduler based on config.

        Args:
            steps_per_epoch: Number of training steps per epoch
        """
        if self.config.scheduler_type == "cosine":
            if self.config.cosine_t_max is None:
                t_max = self.config.num_epochs * steps_per_epoch
            else:
                t_max = self.config.cosine_t_max

            return CosineAnnealingLR(
                self.optimizer, T_max=t_max, eta_min=self.config.cosine_eta_min
            )

        elif self.config.scheduler_type == "none":
            return None

        else:
            raise ValueError(f"Unknown scheduler type: '{self.config.scheduler_type}'")

    def _get_base_criterion(self):
        """Return the loss criterion."""
        return self.criterion

    def _filter_zero_pairs(
        self,
        seq_inputs: torch.Tensor,
        seq_targets: torch.Tensor,
        sample_indices: torch.Tensor,
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Filter out zero signal pairs from batch.

        Sample index 0 represents a zero-signal pair which causes division by ~0
        in relative L2 calculations (||target|| ≈ 0). This is consistently filtered
        in training, validation, and test evaluation loops.

        Args:
            seq_inputs: Input sequences [B, C, T]
            seq_targets: Target sequences [B, C, T]
            sample_indices: Sample indices [B]

        Returns:
            Tuple of (filtered_inputs, filtered_targets, filtered_indices)
            or None if all samples are zero pairs (caller should skip batch)
        """
        non_zero_mask = sample_indices != 0
        if non_zero_mask.sum() == 0:
            return None
        return (
            seq_inputs[non_zero_mask],
            seq_targets[non_zero_mask],
            sample_indices[non_zero_mask],
        )

    def train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch with dual-batch training.

        For DeepONet:
        - Alternates between per-timestep (field) and sequence (BSP) batches
        - Computes field_loss from per-timestep predictions
        - Computes bsp_loss from full-sequence predictions
        - Combines: loss = w_field * field_loss + w_bsp * bsp_loss

        For FNO/CNO:
        - Uses only sequence loader
        - Computes loss from full-sequence predictions

        Returns:
            Dictionary with training metrics:
            - 'loss': Average training loss (combined field + BSP)
            - 'field_loss': Average field loss (DeepONet only)
            - 'bsp_loss': Average BSP loss (DeepONet only)
        """
        self.model.train()

        total_loss = 0.0
        total_field_loss = 0.0
        total_bsp_loss = 0.0
        num_batches = 0

        base_criterion = self._get_base_criterion()

        uses_bsp = getattr(base_criterion, "use_bsp", False)

        use_dual_batch = (
            self.is_deeponet and self.per_timestep_train_loader is not None and uses_bsp
        )

        use_per_timestep_only = (
            self.is_deeponet
            and self.per_timestep_train_loader is not None
            and not uses_bsp
        )

        if use_dual_batch:
            assert (
                self.sequence_train_loader is not None
            ), "sequence_train_loader required for dual-batch"
            assert (
                self.per_timestep_train_loader is not None
            ), "per_timestep_train_loader required for dual-batch"

            for batch_idx, sequence_batch in enumerate(self.sequence_train_loader):
                if len(sequence_batch) == 3:
                    seq_inputs, seq_targets, sample_indices = sequence_batch
                    seq_inputs = seq_inputs.to(self.device)  # [B, 1, 4000]
                    seq_targets = seq_targets.to(self.device)  # [B, 1, 4000]
                    sample_indices = sample_indices.to(
                        self.device
                    )  # [B] sequence sample indices
                else:
                    raise RuntimeError(
                        "Sequence loader must return (input, target, index) tuple. "
                        "Ensure CDONDataset is in 'sequence' mode and returns 3 values."
                    )

                filtered = self._filter_zero_pairs(
                    seq_inputs, seq_targets, sample_indices
                )
                if filtered is None:
                    continue
                seq_inputs, seq_targets, sample_indices = filtered
                earthquake_indices = sample_indices - 1

                assert self.per_timestep_train_dataset is not None
                per_timestep_batch = cast(
                    "CDONDataset", self.per_timestep_train_dataset
                ).get_all_timesteps_for_earthquakes(
                    earthquake_indices=earthquake_indices
                )

                per_ts_inputs = per_timestep_batch["input"].to(
                    self.device
                )  # [B*T_per_eq, 4000]
                per_ts_targets = per_timestep_batch["target"].to(
                    self.device
                )  # [B*T_per_eq]
                per_ts_time_coords = per_timestep_batch["time_coord"].to(
                    self.device
                )  # [B*T_per_eq]

                batch_size_sequence = seq_inputs.shape[0]
                T_per_eq = per_ts_inputs.shape[0] // batch_size_sequence
                if hasattr(self.model, "forward_per_timestep_batched"):
                    unique_time_coords = per_ts_time_coords[:T_per_eq]
                    per_ts_outputs = self.model.forward_per_timestep_batched(
                        per_ts_inputs,
                        n_earthquakes=batch_size_sequence,
                        timesteps_per_earthquake=T_per_eq,
                        time_coords=unique_time_coords,
                    )
                else:
                    per_ts_outputs = self.model.forward_per_timestep(
                        per_ts_inputs, per_ts_time_coords
                    )
                per_ts_outputs = per_ts_outputs.squeeze(
                    -1
                )  # [B*T_per_eq, 1] → [B*T_per_eq]

                if self._check_outputs_for_instability(
                    per_ts_outputs, per_ts_inputs, self.current_epoch, batch_idx
                ):
                    raise RuntimeError(
                        f"Model instability in per-timestep forward at epoch {self.current_epoch}, batch {batch_idx}"
                    )

                field_loss = torch.norm(per_ts_targets - per_ts_outputs, p=2)

                if self._check_for_nan(
                    field_loss, "field_loss", self.current_epoch, batch_idx
                ):
                    raise RuntimeError(
                        f"NaN detected in field loss at epoch {self.current_epoch}, batch {batch_idx}"
                    )

                seq_outputs = self.model(seq_inputs)  # [B, 1, 4000]

                bsp_loss = compute_bsp_loss(
                    seq_outputs, seq_targets, lx=LX_CDON, reduction="sum"
                )

                if self._check_for_nan(
                    bsp_loss, "bsp_loss", self.current_epoch, batch_idx
                ):
                    raise RuntimeError(
                        f"NaN detected in BSP loss at epoch {self.current_epoch}, batch {batch_idx}"
                    )

                self.optimizer.zero_grad()
                combined_loss = self.w_field * field_loss + self.w_bsp * bsp_loss

                combined_loss.backward()

                if self.config.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.max_grad_norm
                    )
                self.optimizer.step()

                if (
                    self.config.scheduler_type == "cosine"
                    and self.scheduler is not None
                ):
                    self.scheduler.step()

                total_loss += combined_loss.item()
                total_field_loss += field_loss.item()
                total_bsp_loss += bsp_loss.item()
                num_batches += 1

                if batch_idx % 10 == 0 and batch_idx > 0:
                    if self._check_model_parameters(self.current_epoch, batch_idx):
                        raise RuntimeError(
                            f"Model parameters corrupted at epoch {self.current_epoch}, batch {batch_idx}"
                        )

        elif use_per_timestep_only:
            assert (
                self.per_timestep_train_loader is not None
            ), "per_timestep_train_loader required for per-timestep-only"

            for batch_idx, per_timestep_batch in enumerate(
                self.per_timestep_train_loader
            ):
                per_ts_inputs = per_timestep_batch["input"].to(self.device)  # [B, 4000]
                per_ts_targets = per_timestep_batch["target"].to(self.device)  # [B]
                per_ts_time_coords = per_timestep_batch["time_coord"].to(
                    self.device
                )  # [B]

                self.optimizer.zero_grad()

                per_ts_outputs = self.model.forward_per_timestep(
                    per_ts_inputs, per_ts_time_coords
                )
                per_ts_outputs = per_ts_outputs.squeeze(-1)  # [B, 1] → [B]

                if self._check_outputs_for_instability(
                    per_ts_outputs, per_ts_inputs, self.current_epoch, batch_idx
                ):
                    raise RuntimeError(
                        f"Model instability detected at epoch {self.current_epoch}, batch {batch_idx}. "
                        "Outputs contain Inf/NaN. See diagnostics above."
                    )

                final_loss = torch.norm(per_ts_targets - per_ts_outputs, p=2)

                if self._check_for_nan(
                    final_loss, "field_loss", self.current_epoch, batch_idx
                ):
                    raise RuntimeError(
                        f"NaN detected in field loss at epoch {self.current_epoch}, batch {batch_idx}"
                    )

                final_loss.backward()
                if self.config.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.max_grad_norm
                    )
                self.optimizer.step()

                if (
                    self.config.scheduler_type == "cosine"
                    and self.scheduler is not None
                ):
                    self.scheduler.step()

                total_loss += final_loss.item()
                total_field_loss += final_loss.item()
                total_bsp_loss += 0.0
                num_batches += 1

                if batch_idx % 10 == 0 and batch_idx > 0:
                    if self._check_model_parameters(self.current_epoch, batch_idx):
                        raise RuntimeError(
                            f"Model parameters corrupted at epoch {self.current_epoch}, batch {batch_idx}. "
                            "Reduce learning rate and restart from checkpoint."
                        )

        else:
            assert (
                self.sequence_train_loader is not None
            ), "sequence_train_loader required for sequence-only training"

            for batch_idx, batch in enumerate(self.sequence_train_loader):
                if len(batch) == 3:
                    seq_inputs, seq_targets, sample_indices = batch
                    seq_inputs = seq_inputs.to(self.device)  # [B, 1, 4000]
                    seq_targets = seq_targets.to(self.device)  # [B, 1, 4000]
                    sample_indices = sample_indices.to(self.device)  # [B]
                else:
                    seq_inputs = batch[0].to(self.device)
                    seq_targets = batch[1].to(self.device)
                    sample_indices = None

                if sample_indices is not None:
                    filtered = self._filter_zero_pairs(
                        seq_inputs, seq_targets, sample_indices
                    )
                    if filtered is None:
                        continue
                    seq_inputs, seq_targets, sample_indices = filtered

                self.optimizer.zero_grad()

                seq_outputs = self.model(seq_inputs)  # [B, 1, 4000]

                diff = seq_targets - seq_outputs  # [B, 1, T]
                field_loss = torch.norm(diff, p=2)

                if getattr(base_criterion, "use_bsp", False):
                    bsp_loss = compute_bsp_loss(seq_outputs, seq_targets, lx=LX_CDON)
                    final_loss = field_loss + self.w_bsp * bsp_loss
                    field_component = field_loss
                    bsp_component = bsp_loss
                else:
                    final_loss = field_loss
                    field_component = field_loss
                    bsp_component = torch.tensor(0.0, device=self.device)

                if self._check_for_nan(
                    final_loss, "sequence_loss", self.current_epoch, batch_idx
                ):
                    raise RuntimeError(
                        f"NaN detected in sequence loss at epoch {self.current_epoch}, batch {batch_idx}"
                    )

                final_loss.backward()
                if self.config.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.max_grad_norm
                    )
                self.optimizer.step()

                if (
                    self.config.scheduler_type == "cosine"
                    and self.scheduler is not None
                ):
                    self.scheduler.step()

                total_loss += final_loss.item()
                total_field_loss += field_component.item()
                total_bsp_loss += bsp_component.item()
                num_batches += 1

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0

        metrics = {"loss": avg_loss}

        if self.is_deeponet:
            metrics["field_loss"] = (
                total_field_loss / num_batches if num_batches > 0 else 0.0
            )
            metrics["bsp_loss"] = (
                total_bsp_loss / num_batches if num_batches > 0 else 0.0
            )

        return metrics

    @torch.no_grad()
    def evaluate_test(self) -> Dict[str, float]:
        """
        Evaluate on test set and compute NRMSE for model selection.

        This is used for test-based best model selection. Uses field NRMSE
        (per-earthquake min-max normalized RMSE) as the primary selection metric.

        Returns:
            Dictionary with test metrics:
            - 'test_nrmse': Field nRMSE (mean across all samples)
            - 'test_l2': Relative L2 error (mean across all samples)
            - 'test_mse': Mean squared error
            - 'test_samples': Number of samples evaluated
            - 'filtered_samples': Number of zero signal pairs filtered out
        """
        if self.test_loader is None:
            return {
                "test_nrmse": float("inf"),
                "test_l2": float("inf"),
                "test_mse": float("inf"),
            }

        self.model.eval()

        all_outputs = []
        all_targets = []
        total_l2 = 0.0
        total_mse = 0.0
        num_samples = 0
        num_filtered = 0

        with torch.no_grad():
            for batch in self.test_loader:
                if len(batch) == 3:
                    inputs, targets, indices = batch
                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)
                    indices = indices.to(self.device)

                    non_zero_mask = indices != 0
                    num_zero_in_batch = (~non_zero_mask).sum().item()
                    if num_zero_in_batch > 0:
                        num_filtered += num_zero_in_batch
                        if non_zero_mask.sum() == 0:
                            continue
                        inputs = inputs[non_zero_mask]
                        targets = targets[non_zero_mask]
                else:
                    inputs = batch[0].to(self.device)
                    targets = batch[1].to(self.device)

                if self.is_deeponet:
                    if hasattr(self.model, "forward_causal_sequence"):
                        outputs = self.model.forward_causal_sequence(inputs)
                    else:
                        outputs = self.model(inputs)
                else:
                    outputs = self.model(inputs)

                all_outputs.append(outputs.detach())
                all_targets.append(targets.detach())

                outputs_for_error = outputs.squeeze(1)  # [B, T]
                targets_for_error = targets.squeeze(1)  # [B, T]

                for i in range(outputs_for_error.shape[0]):
                    pred = outputs_for_error[i]
                    target = targets_for_error[i]

                    l2_error = torch.sqrt(torch.sum((pred - target) ** 2))
                    l2_norm = torch.sqrt(torch.sum(target**2))
                    rel_l2 = l2_error / (l2_norm + EPSILON)
                    total_l2 += rel_l2.item()

                    total_mse += torch.mean((pred - target) ** 2).item()
                    num_samples += 1

        if all_outputs and all_targets:
            all_outputs_tensor = torch.cat(all_outputs, dim=0)  # [N, C, T]
            all_targets_tensor = torch.cat(all_targets, dim=0)  # [N, C, T]

            field_nrmse = compute_field_nrmse(all_outputs_tensor, all_targets_tensor)
            avg_nrmse = field_nrmse.mean().item()
        else:
            avg_nrmse = float("inf")

        if num_filtered > 0 and self.config.verbose:
            self.console.print(
                f"  [dim]Filtered {num_filtered} zero signal pair(s) from test evaluation[/dim]"
            )

        avg_l2 = total_l2 / max(num_samples, 1)
        avg_mse = total_mse / max(num_samples, 1)

        return {
            "test_nrmse": avg_nrmse,
            "test_l2": avg_l2,
            "test_mse": avg_mse,
            "test_samples": num_samples,
            "filtered_samples": num_filtered,
        }

    def save_checkpoint(
        self,
        epoch: int,
        metrics: Dict[str, float],
        is_best: bool = False,
        is_latest: bool = False,
    ) -> None:
        """
        Save model checkpoint.

        Args:
            epoch: Current epoch number
            metrics: Validation metrics dictionary
            is_best: Whether this is the best model so far
            is_latest: Whether this is the latest model
        """
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": (
                self.scheduler.state_dict() if self.scheduler else None
            ),
            "config": self.config.to_dict(),
            "metrics": metrics,
            "best_test_nrmse": self.best_test_nrmse,
            "experiment_name": self.experiment_name,
        }

        if is_best:
            best_path = self.checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            if self.config.verbose:
                if "test_nrmse" in metrics:
                    self.console.print(
                        f"[bold green]✓[/bold green] Saved best model "
                        f"(test_nrmse: {metrics['test_nrmse']:.4f})"
                    )
                else:
                    self.console.print("[bold green]✓[/bold green] Saved best model")

        if is_latest:
            latest_path = self.checkpoint_dir / "latest_model.pt"
            torch.save(checkpoint, latest_path)

        milestone_epochs = self.config.save_milestone_epochs
        milestone_freq = self.config.save_milestone_frequency
        max_milestone = max(milestone_epochs) if milestone_epochs else 0

        is_milestone = epoch in milestone_epochs or (
            epoch > max_milestone and milestone_freq > 0 and epoch % milestone_freq == 0
        )

        if is_milestone:
            milestone_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
            torch.save(checkpoint, milestone_path)
            if self.config.verbose:
                self.console.print(
                    f"[bold blue]📌[/bold blue] Saved milestone checkpoint (epoch {epoch})"
                )

    def train(self) -> Dict[str, Any]:
        """
        Main training loop.

        Returns:
            Dictionary with training history:
            - 'train_history': List of training metrics per epoch
            - 'test_history': List of test metrics per epoch
            - 'best_test_nrmse': Best test nRMSE achieved
            - 'final_epoch': Final epoch number
        """
        if self.config.verbose:
            self.console.print(f"\n[bold]Training {self.experiment_name}[/bold]")
            self.console.print(f"Device: {self.device}")
            self.console.print(
                f"Model: {'DeepONet' if self.is_deeponet else 'FNO/CNO'}"
            )
            self.console.print(
                f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}"
            )

            has_dual_loaders = (
                self.per_timestep_train_loader is not None
                and self.sequence_train_loader is not None
            )
            has_per_timestep_only = (
                self.per_timestep_train_loader is not None
                and self.sequence_train_loader is None
            )

            if self.is_deeponet and has_dual_loaders:
                self.console.print(
                    f"Training samples (per-timestep): {len(self.per_timestep_train_loader.dataset):,}"
                )
                self.console.print(
                    f"Training samples (sequences): {len(self.sequence_train_loader.dataset):,}"
                )
            elif self.is_deeponet and has_per_timestep_only:
                self.console.print(
                    f"Training samples (per-timestep): {len(self.per_timestep_train_loader.dataset):,}"
                )
            else:
                self.console.print(
                    f"Training samples (sequences): {len(self.sequence_train_loader.dataset):,}"
                )

            self.console.print(
                f"Loss weights: λ_field={self.w_field}, λ_bsp={self.w_bsp}\n"
            )

        if self.is_deeponet:
            progress_columns = [
                TextColumn("[bold cyan]Epoch {task.fields[epoch]}/{task.total}"),
                BarColumn(),
                TextColumn("•"),
                TextColumn("Loss: {task.fields[train_loss]:.4f}"),
                TextColumn(
                    "[dim](Field: {task.fields[train_field]:.4f} BSP: {task.fields[train_bsp]:.4f})[/dim]"
                ),
                TextColumn("•"),
                TextColumn("Test: {task.fields[test_loss]:.4f}"),
                TextColumn("•"),
                TextColumn("LR: {task.fields[lr]:.2e}"),
                TimeElapsedColumn(),
            ]
            initial_task_fields = {
                "epoch": 0,
                "train_loss": 0.0,
                "train_field": 0.0,
                "train_bsp": 0.0,
                "test_loss": 0.0,
                "lr": self.config.learning_rate,
            }
        else:
            progress_columns = [
                TextColumn("[bold cyan]Epoch {task.fields[epoch]}/{task.total}"),
                BarColumn(),
                TextColumn("•"),
                TextColumn("Train Loss: {task.fields[train_loss]:.4f}"),
                TextColumn("•"),
                TextColumn("Test Loss: {task.fields[test_loss]:.4f}"),
                TextColumn("•"),
                TextColumn("LR: {task.fields[lr]:.2e}"),
                TimeElapsedColumn(),
            ]
            initial_task_fields = {
                "epoch": 0,
                "train_loss": 0.0,
                "test_loss": 0.0,
                "lr": self.config.learning_rate,
            }

        start_time = time.time()
        stopped_early = False

        # Main epoch loop with progress bar
        with Progress(*progress_columns) as progress:

            epoch_task = progress.add_task(
                "Training", total=self.config.num_epochs, **initial_task_fields
            )

            for epoch in range(1, self.config.num_epochs + 1):
                if (
                    self.config.max_training_time is not None
                    and self.config.max_training_time > 0
                ):
                    elapsed = time.time() - start_time
                    if elapsed >= self.config.max_training_time:
                        self.console.print(
                            f"\n[bold yellow]⏰ Time limit ({self.config.max_training_time/60:.1f} min) "
                            f"reached at epoch {epoch-1}.[/bold yellow]"
                        )
                        self.console.print(
                            f"   Elapsed: {elapsed/60:.1f} min. Stopping training."
                        )
                        stopped_early = True
                        break

                self.current_epoch = epoch

                train_metrics = self.train_epoch()

                test_metrics = {"test_nrmse": float("inf"), "test_l2": float("inf")}
                if self.has_test_selection and epoch % self.config.eval_frequency == 0:
                    test_metrics = self.evaluate_test()
                    test_metrics["epoch"] = epoch
                    self.test_history.append(test_metrics)

                is_best = False
                if self.has_test_selection:
                    if test_metrics["test_nrmse"] < self.best_test_nrmse:
                        is_best = True
                        self.best_test_nrmse = test_metrics["test_nrmse"]

                if self.config.save_best and is_best:
                    self.save_checkpoint(epoch, test_metrics, is_best=True)

                if self.config.save_latest:
                    self.save_checkpoint(epoch, test_metrics, is_latest=True)

                self.train_history.append(train_metrics)

                current_lr = self.optimizer.param_groups[0]["lr"]
                update_fields = {
                    "epoch": epoch,
                    "train_loss": train_metrics["loss"],
                    "test_loss": test_metrics.get("test_l2", 0.0),
                    "lr": current_lr,
                }

                if self.is_deeponet:
                    update_fields["train_field"] = train_metrics.get("field_loss", 0.0)
                    update_fields["train_bsp"] = train_metrics.get("bsp_loss", 0.0)

                progress.update(epoch_task, advance=1, **update_fields)

        total_training_time = time.time() - start_time

        if self.config.verbose:
            if stopped_early:
                self.console.print(
                    "\n[bold yellow]Training stopped early (time limit).[/bold yellow]"
                )
            else:
                self.console.print("\n[bold green]Training complete![/bold green]")
            self.console.print(
                f"Total time: {total_training_time/60:.1f} min ({self.current_epoch} epochs)"
            )
            if self.has_test_selection:
                self.console.print(f"Best test nRMSE: {self.best_test_nrmse:.4f}")
            self.console.print(f"Checkpoints saved to: {self.checkpoint_dir}")

        return {
            "train_history": self.train_history,
            "test_history": self.test_history,
            "best_test_nrmse": self.best_test_nrmse,
            "final_epoch": self.current_epoch,
            "stopped_early": stopped_early,
            "total_training_time": total_training_time,
        }
