"""
PyTorch Dataset class for CDON earthquake data.

Loads CDON data (earthquake accelerations → structural displacements)
and formats it as expected by the models.

Supports causal zero-padding preprocessing to enforce physical causality at the data level.
"""

import os
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from src.core.evaluation.constants import SIGNAL_LENGTH_CDON
from src.data.preprocessing_utils import prepare_causal_data
from torch.utils.data import Dataset


class CDONDataset(Dataset):
    """
    PyTorch Dataset for CDON earthquake data.

    Supports two modes:
    1. 'per_timestep': Sliding windows with zero-padding for DeepONet causal loss
       - Returns: {'input': [4000], 'target': [], 'time_coord': []}
       - Total samples: n_earthquakes × (signal_length + 1) (includes initial condition)
       - Example: 80 earthquakes × 4001 = 320,080 samples
       - Prediction at timestep t only uses input [0, ..., t] via zero-padding.

    2. 'sequence': Full sequences without padding for BSP loss and FNO/CNO field
       - Returns: (input [1, 4000], target [1, 4000])
       - Total samples: n_earthquakes + 1 (includes zero signal pair for homogeneity)

    Initial Condition Enforcement:
        In 'per_timestep' mode, includes an explicit sample at t=0 with input=zeros
        and target=0 to enforce that displacement is 0 at the initial time instant.

    """

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        val_split_ratio: float = 0.0,
        val_split_seed: int = 42,
        signal_length: int = SIGNAL_LENGTH_CDON,
        mode: str = "per_timestep",
        target_signal_length: Optional[int] = None,
    ):
        """Initialize CDON data in per-timestep or sequence mode."""
        self.data_dir = data_dir
        self.split = split
        self.val_split_ratio = val_split_ratio
        self.val_split_seed = val_split_seed
        self.signal_length = signal_length
        self.mode = mode
        self.target_signal_length = target_signal_length

        if split not in ["train", "val", "test"]:
            raise ValueError(
                f"Invalid split '{split}'. Must be 'train', 'val', or 'test'."
            )
        if mode not in ["per_timestep", "sequence"]:
            raise ValueError(
                f"Invalid mode '{mode}'. Must be 'per_timestep' or 'sequence'."
            )

        self.loads, self.responses = self._load_data()

        if self.loads.shape != self.responses.shape:
            raise ValueError(
                f"Shape mismatch: loads {self.loads.shape} != responses {self.responses.shape}"
            )

        self.n_earthquakes, self.n_timesteps = self.loads.shape

        if mode == "per_timestep":
            self._prepare_windowed_data()
        else:
            self._prepare_sequence_data()

    def _load_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load appropriate data based on split.

        For 'train' and 'val': Load train_*.npy files, then split
        For 'test': Load test_*.npy files

        Returns:
            Tuple of (loads, responses) arrays with shape [n_samples, n_timesteps]
        """
        if self.split in ["train", "val"]:
            loads_path = os.path.join(self.data_dir, "train_Loads.npy")
            responses_path = os.path.join(self.data_dir, "train_Responses.npy")

            if not os.path.exists(loads_path):
                raise FileNotFoundError(f"Train loads not found: {loads_path}")
            if not os.path.exists(responses_path):
                raise FileNotFoundError(f"Train responses not found: {responses_path}")

            loads = np.load(loads_path)
            responses = np.load(responses_path)

            if self.val_split_ratio == 0.0:
                if self.split == "val":
                    raise ValueError(
                        "Cannot create 'val' split with val_split_ratio=0.0. "
                        "All training data is used for training (no validation set)."
                    )
            else:
                n_samples = loads.shape[0]
                n_val = int(n_samples * self.val_split_ratio)
                n_train = n_samples - n_val

                rng = np.random.RandomState(self.val_split_seed)
                indices = np.arange(n_samples)
                rng.shuffle(indices)

                if self.split == "train":
                    selected_indices = indices[:n_train]
                else:  # val
                    selected_indices = indices[n_train:]

                loads = loads[selected_indices]
                responses = responses[selected_indices]

        else:
            loads_path = os.path.join(self.data_dir, "test_Loads.npy")
            responses_path = os.path.join(self.data_dir, "test_Responses.npy")

            if not os.path.exists(loads_path):
                raise FileNotFoundError(f"Test loads not found: {loads_path}")
            if not os.path.exists(responses_path):
                raise FileNotFoundError(f"Test responses not found: {responses_path}")

            loads = np.load(loads_path)
            responses = np.load(responses_path)

        return loads, responses

    def _prepare_windowed_data(self):
        """
        Convert raw earthquake data to per-timestep windowed samples.

        Uses prepare_causal_data() to create sliding windows with zero-padding.
        Also computes time coordinates.

        Initial Condition Enforcement:
            Includes an explicit sample at t=0 with input=zeros and target=0 to enforce
            that displacement is 0.

        Sets:
            self.windowed_inputs: [N*(T+1), signal_length] - zero-padded windows (includes initial condition)
            self.windowed_targets: [N*(T+1)] - scalar outputs (includes initial condition = 0)
            self.time_coords: [N*(T+1)] - time coordinates in [0, 1] (includes t=0)
            self.samples_per_earthquake: T+1 - number of samples per earthquake (includes initial condition)
        """
        loads_tensor = torch.from_numpy(self.loads).float()  # [N, T]
        responses_tensor = torch.from_numpy(self.responses).float()  # [N, T]

        # Create windowed causal data with initial condition enforcement
        # prepare_causal_data converts [N, T] → [N*(T+1), T] inputs and [N*(T+1)] outputs
        self.windowed_inputs, self.windowed_targets = prepare_causal_data(
            loads_tensor, responses_tensor, signal_length=self.signal_length
        )

        # (includes initial condition)
        self.samples_per_earthquake = self.signal_length + 1  # T+1

        time_grid = torch.linspace(0, 1, self.samples_per_earthquake)  # [T+1]
        self.time_coords = time_grid.repeat(self.n_earthquakes)  # [N*(T+1)]

        self.n_samples = self.windowed_inputs.shape[0]  # N * (T+1)

    def _prepare_sequence_data(self):
        """
        Prepare full sequences for BSP loss and sequence-based models.

        Creates full sequences without padding for FNO, CNO, and BSP loss computation.
        Shape: [N+1, 1, signal_length] (includes zero signal pair for homogeneity constraint)

        Homogeneity constraint:
            Adds a zero signal pair (0,0) to enforce zero input → zero output.

        Sets:
            self.sequence_inputs: [N+1, 1, signal_length] full input sequences (with zero pair)
            self.sequence_targets: [N+1, 1, signal_length] full target sequences (with zero pair)
            self.n_samples: N+1 - number of earthquakes plus zero pair
        """
        loads_tensor = torch.from_numpy(self.loads).float()  # [N, T]
        responses_tensor = torch.from_numpy(self.responses).float()  # [N, T]

        # Add zero signal pair for homogeneity constraint.
        # Physical meaning: zero earthquake acceleration → zero displacement.
        zero_input = torch.zeros(1, self.signal_length)  # [1, T] - all zeros
        zero_output = torch.zeros(1, self.signal_length)  # [1, T] - all zeros

        loads_tensor = torch.cat([zero_input, loads_tensor], dim=0)  # [N+1, T]
        responses_tensor = torch.cat([zero_output, responses_tensor], dim=0)  # [N+1, T]
        n_samples_with_zero = loads_tensor.shape[0]  # N+1

        self.sequence_inputs = loads_tensor.unsqueeze(1)  # [N+1, 1, signal_length]
        self.sequence_targets = responses_tensor.unsqueeze(1)  # [N+1, 1, signal_length]

        if (
            self.target_signal_length is not None
            and self.target_signal_length != self.signal_length
        ):
            import torch.nn.functional as F

            self.sequence_inputs = F.interpolate(
                self.sequence_inputs,
                size=self.target_signal_length,
                mode="linear",
                align_corners=True,
            )
            self.sequence_targets = F.interpolate(
                self.sequence_targets,
                size=self.target_signal_length,
                mode="linear",
                align_corners=True,
            )
            self._output_signal_length = self.target_signal_length
        else:
            self._output_signal_length = self.signal_length

        self.n_samples = n_samples_with_zero

    def __len__(self) -> int:
        """
        Return number of samples in dataset.

        Returns:
            For 'per_timestep' mode: n_earthquakes × (signal_length + 1) (e.g., 320,080 for 80 earthquakes)
            For 'sequence' mode: n_earthquakes + 1 (includes zero signal pair for homogeneity)
        """
        return self.n_samples

    def __getitem__(self, idx: int):
        """
        Get a single sample from dataset.

        Args:
            idx: Sample index

        Returns:
            For 'per_timestep' mode:
                Dictionary with keys:
                - 'input': [signal_length] - windowed input with zero-padding
                - 'target': [] - scalar response at timestep (0 for initial condition)
                - 'time_coord': [] - scalar time in [0, 1]
                - 'sample_idx': int - earthquake index

            For 'sequence' mode:
                Tuple: (input [1, 4000], target [1, 4000], idx: int) - full sequences + index
        """
        if self.mode == "per_timestep":
            sample_idx = idx // self.samples_per_earthquake
            return {
                "input": self.windowed_inputs[idx],  # [signal_length]
                "target": self.windowed_targets[idx],  # [] scalar
                "time_coord": self.time_coords[idx],  # [] scalar
                "sample_idx": sample_idx,  # Earthquake index
            }
        else:  # mode == 'sequence'
            return (self.sequence_inputs[idx], self.sequence_targets[idx], idx)

    def get_all_timesteps_for_earthquakes(
        self, earthquake_indices: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Get all per-timestep samples for selected earthquakes.

        Returns:
            - input: [B * (signal_length+1), signal_length]
            - target: [B * (signal_length+1)]
            - time_coord: [B * (signal_length+1)]
            - sample_idx: [B * (signal_length+1)]
        """
        if self.mode != "per_timestep":
            raise RuntimeError(
                "get_all_timesteps_for_earthquakes() requires mode='per_timestep'. "
                f"Current mode: '{self.mode}'"
            )

        earthquake_indices = earthquake_indices.cpu()
        max_idx = earthquake_indices.max().item()
        min_idx = earthquake_indices.min().item()
        if min_idx < 0:
            raise IndexError(
                f"Negative earthquake index {min_idx} not allowed. "
                f"Valid range: [0, {self.n_earthquakes})"
            )
        if max_idx >= self.n_earthquakes:
            raise IndexError(
                f"Earthquake index {max_idx} out of range [0, {self.n_earthquakes})"
            )

        batch_size = earthquake_indices.shape[0]
        total_samples = batch_size * self.samples_per_earthquake

        inputs = torch.zeros(total_samples, self.signal_length)
        targets = torch.zeros(total_samples)
        time_coords = torch.zeros(total_samples)
        sample_indices = torch.zeros(total_samples, dtype=torch.long)

        for i, eq_idx in enumerate(earthquake_indices):
            eq_idx = eq_idx.item()

            base_idx = eq_idx * self.samples_per_earthquake
            flat_indices = torch.arange(
                base_idx, base_idx + self.samples_per_earthquake
            )

            out_start = i * self.samples_per_earthquake
            out_end = out_start + self.samples_per_earthquake

            inputs[out_start:out_end] = self.windowed_inputs[flat_indices]
            targets[out_start:out_end] = self.windowed_targets[flat_indices]
            time_coords[out_start:out_end] = self.time_coords[flat_indices]
            sample_indices[out_start:out_end] = eq_idx

        return {
            "input": inputs,
            "target": targets,
            "time_coord": time_coords,
            "sample_idx": sample_indices,
        }
