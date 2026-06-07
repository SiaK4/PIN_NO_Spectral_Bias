"""
Base class for DeepO architectures (DeepONet, DeepOKAN).

Implements shared branch-trunk forward methods with architecture-specific
output layer handling via abstract method.

Both DeepONet and DeepOKAN share the same branch-trunk architecture:
- Branch network: Encodes input function → latent vector
- Trunk network: Encodes time coordinate → latent vector
- Combination: Element-wise product + output layer → scalar

Differences include the network type (MLP vs KAN) and output layer handling.
"""

from abc import ABC, abstractmethod
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseDeepO(nn.Module, ABC):
    """
    Abstract base class for DeepO architectures (DeepONet, DeepOKAN).

    Provides shared branch-trunk forward methods (per-timestep for field, full-sequence
    for BSP, causal-sequence with causality enforcement, plus GPU-aware chunking).

    Subclasses must:
    1. Define in __init__: self.branch, self.trunk, self.sensor_dim, self.latent_dim
    2. Implement: _apply_output_layer(combined, is_batched_3d)
    """

    supports_per_timestep = True

    # Attributes subclasses must define
    branch: nn.Module
    trunk: nn.Module
    sensor_dim: int
    latent_dim: int

    @abstractmethod
    def _apply_output_layer(
        self, combined: torch.Tensor, is_batched_3d: bool = False
    ) -> torch.Tensor:
        """
        Apply architecture-specific output layer.

        Args:
            combined: Combined branch*trunk tensor
                     - 2D: [batch, latent_dim] for per-timestep modes
                     - 3D: [batch, seq_len, latent_dim] for full-sequence (is_batched_3d=True)
            is_batched_3d: Whether input is 3D (full-sequence mode)

        Returns:
            Output tensor:
            - 2D input: [batch, 1]
            - 3D input: [batch, 1, seq_len] (note: transposed for consistency)
        """
        pass

    def forward_per_timestep(
        self, x: torch.Tensor, time_coord: torch.Tensor
    ) -> torch.Tensor:
        """
        Per-timestep forward pass for field loss.

        Args:
            x: Windowed input tensor of shape [batch, sensor_dim]
               Each sample is a zero-padded causal window
            time_coord: Time coordinates of shape [batch, 1] or [batch]
                       Normalized time values in [0, 1]

        Returns:
            Output tensor of shape [batch, 1] - scalar prediction per sample

        Example:
            >>> model = DeepONet(sensor_dim=4000, latent_dim=100)
            >>> x = torch.randn(16, 4000)  # 16 windowed samples
            >>> t = torch.rand(16, 1)      # 16 time coordinates
            >>> y = model.forward_per_timestep(x, t)  # Output: [16, 1]
        """
        # Ensure time_coord has shape [batch, 1]
        if time_coord.ndim == 1:
            time_coord = time_coord.unsqueeze(-1)  # [batch] → [batch, 1]

        # [batch, sensor_dim] → [batch, latent_dim]
        branch_output = self.branch(x)

        # [batch, 1] → [batch, latent_dim]
        trunk_output = self.trunk(time_coord)

        combined = branch_output * trunk_output  # [batch, latent_dim]

        return self._apply_output_layer(combined, is_batched_3d=False)  # [batch, 1]

    def forward_per_timestep_batched(
        self,
        inputs: torch.Tensor,
        n_earthquakes: int,
        timesteps_per_earthquake: int,
        time_coords: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Batched per-timestep forward with trunk caching.

        inputs are laid out as B contiguous blocks of T causal windows:
        [B*T, sensor_dim] -> [B*T, 1].
        """
        device = inputs.device
        dtype = inputs.dtype
        B = n_earthquakes
        T = timesteps_per_earthquake

        expected_samples = B * T
        if inputs.shape[0] != expected_samples:
            raise ValueError(
                f"Expected {expected_samples} samples (B={B} × T={T}), "
                f"got {inputs.shape[0]}"
            )

        if time_coords is None:
            time_coords = torch.linspace(0, 1, T, device=device, dtype=dtype)

        time_input = time_coords.unsqueeze(-1)  # [T, 1]

        trunk_output = self.trunk(time_input)  # [T, latent_dim]
        branch_output = self.branch(inputs)  # [B*T, latent_dim]

        timestep_indices = torch.arange(T, device=device).repeat(B)  # [B*T]
        trunk_indexed = trunk_output[timestep_indices]  # [B*T, latent_dim]

        combined = branch_output * trunk_indexed  # [B*T, latent_dim]

        return self._apply_output_layer(combined, is_batched_3d=False)  # [B*T, 1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Full-sequence forward pass for BSP loss (no causality constraint).

        Args:
            x: Full sequence input tensor of shape [batch, 1, sensor_dim]
               Raw signals without zero-padding

        Returns:
            Output tensor of shape [batch, 1, sensor_dim] - full sequence predictions

        Optimization:
            Trunk is computed once for the sensor_dim time coordinates (not batch*sensor_dim).
            Since time grid is identical across batch, we compute trunk once and broadcast.
        """
        x_flat = x.squeeze(1)  # [batch, sensor_dim]

        # [batch, sensor_dim] → [batch, latent_dim]
        branch_output = self.branch(x_flat)

        time_grid = torch.linspace(
            0, 1, self.sensor_dim, device=x.device
        )  # [sensor_dim]
        time_input = time_grid.unsqueeze(-1)  # [sensor_dim, 1]

        trunk_output = self.trunk(time_input)  # [sensor_dim, latent_dim]

        trunk_expanded = trunk_output.unsqueeze(0)  # [1, sensor_dim, latent_dim]
        branch_expanded = branch_output.unsqueeze(1)  # [batch, 1, latent_dim]

        combined = branch_expanded * trunk_expanded  # [batch, sensor_dim, latent_dim]

        return self._apply_output_layer(
            combined, is_batched_3d=True
        )  # [batch, 1, sensor_dim]

    def _estimate_chunk_size(self, batch_size: int, device: torch.device) -> int:
        """
        Estimate optimal chunk size based on available GPU memory.

        The goal is to maximize GPU utilization without running out of memory.
        We estimate memory per chunk and pick the largest chunk that fits.

        Args:
            batch_size: Number of earthquakes in batch
            device: Device (cuda or cpu)

        Returns:
            Recommended chunk size (number of timesteps to process together)
        """
        if device.type != "cuda":
            return 200

        try:
            free_memory = torch.cuda.get_device_properties(device).total_memory
            allocated = torch.cuda.memory_allocated(device)
            available = free_memory - allocated

            # Estimate memory per sample in chunk:
            # - Causal window: sensor_dim floats = 4000 * 4 bytes = 16KB
            # - Branch output: latent_dim floats = 100 * 4 bytes = 400B
            # - Intermediate activations: ~3x branch size (conservative for KAN)
            # - Gradients: ~2x forward memory
            bytes_per_sample = (self.sensor_dim + self.latent_dim * 4) * 4 * 3
            bytes_per_chunk_unit = batch_size * bytes_per_sample

            safe_memory = available * 0.5
            max_chunk = int(safe_memory / bytes_per_chunk_unit)

            chunk_size = max(50, min(500, max_chunk))

            return chunk_size

        except Exception:
            return 200

    def forward_causal_sequence(
        self, x: torch.Tensor, chunk_size: Optional[int] = None
    ) -> torch.Tensor:
        """
        Full-sequence prediction with causal windows.

        For timestep t, the branch receives only inputs [0, ..., t], right-aligned
        with left zero-padding. Returns [batch, 1, sensor_dim].
        """
        batch_size = x.shape[0]
        device = x.device
        dtype = x.dtype

        if chunk_size is None:
            chunk_size = self._estimate_chunk_size(batch_size, device)

        # [batch, 1, sensor_dim] → [batch, sensor_dim]
        x_flat = x.squeeze(1)

        time_coords = torch.linspace(
            1 / self.sensor_dim, 1, self.sensor_dim, device=device, dtype=dtype
        )
        time_input = time_coords.unsqueeze(-1)  # [sensor_dim, 1]

        all_trunk_outputs = self.trunk(time_input)  # [sensor_dim, latent_dim]

        # Pad left with zeros: [batch, sensor_dim] → [batch, 2*sensor_dim - 1]
        x_padded = F.pad(x_flat, (self.sensor_dim - 1, 0), mode="constant", value=0.0)

        # Unfold to create all causal windows at once
        # [batch, 2*sensor_dim-1] → [batch, sensor_dim, sensor_dim]
        # Each all_windows[b, t, :] is the causal window for batch b, timestep t
        all_windows = x_padded.unfold(dimension=1, size=self.sensor_dim, step=1)

        chunk_predictions_list = []
        for chunk_start in range(0, self.sensor_dim, chunk_size):
            chunk_end = min(chunk_start + chunk_size, self.sensor_dim)
            actual_chunk_size = chunk_end - chunk_start

            chunk_windows = all_windows[
                :, chunk_start:chunk_end, :
            ]  # [batch, chunk_size, sensor_dim]
            windows_flat = chunk_windows.reshape(
                batch_size * actual_chunk_size, self.sensor_dim
            )  # [batch * chunk_size, sensor_dim]

            branch_outputs = self.branch(
                windows_flat
            )  # [batch * chunk_size, latent_dim]

            # Get pre-computed trunk outputs for this chunk: [chunk_size, latent_dim]
            trunk_chunk = all_trunk_outputs[chunk_start:chunk_end, :]

            branch_reshaped = branch_outputs.reshape(
                batch_size, actual_chunk_size, self.latent_dim
            )  # [batch, chunk_size, latent_dim]

            trunk_expanded = trunk_chunk.unsqueeze(0)  # [1, chunk_size, latent_dim]

            combined = branch_reshaped * trunk_expanded
            combined_flat = combined.reshape(
                batch_size * actual_chunk_size, self.latent_dim
            )  # [batch, chunk_size, latent_dim]
            chunk_output = self._apply_output_layer(combined_flat, is_batched_3d=False)
            chunk_predictions = chunk_output.reshape(batch_size, actual_chunk_size)
            chunk_predictions_list.append(chunk_predictions)

        predictions = torch.cat(chunk_predictions_list, dim=1)  # [batch, sensor_dim]
        output = predictions.unsqueeze(1)

        return output  # [batch, 1, sensor_dim]
