"""
Data preprocessing utilities.
"""

from typing import Tuple, Union

import numpy as np
import torch


def prepare_causal_data(
    inputs: Union[torch.Tensor, np.ndarray],
    outputs: Union[torch.Tensor, np.ndarray],
    signal_length: int = 4000,
) -> Tuple[Union[torch.Tensor, np.ndarray], Union[torch.Tensor, np.ndarray]]:
    """
    Create causal windows with left zero-padding.

    For each timestep t, the window is padded_signal[t : t + signal_length],
    containing inputs[0:t+1] right-aligned. Returns [N*(T+1), T] inputs and
    [N*(T+1)] outputs, including the zero initial-condition sample.
    """
    inputs_is_torch = isinstance(inputs, torch.Tensor)
    outputs_is_torch = isinstance(outputs, torch.Tensor)

    if inputs_is_torch:
        inputs_np = inputs.cpu().numpy()
    else:
        inputs_np = inputs

    if outputs_is_torch:
        outputs_np = outputs.cpu().numpy()
    else:
        outputs_np = outputs

    if inputs_np.ndim == 3:
        inputs_np = inputs_np[:, 0, :]
    if outputs_np.ndim == 3:
        outputs_np = outputs_np[:, 0, :]

    n_samples = inputs_np.shape[0]

    samples_per_earthquake = signal_length + 1  # +1 for initial condition
    causal_inputs_list = []
    causal_outputs_list = []

    for idx in range(n_samples):
        padded_length = signal_length - 1 + signal_length
        zero_padded_input = np.zeros(padded_length, dtype=np.float32)
        zero_padded_input[signal_length - 1 :] = inputs_np[idx]

        windowed_inputs = np.zeros(
            (samples_per_earthquake, signal_length), dtype=np.float32
        )
        timestep_outputs = np.zeros(samples_per_earthquake, dtype=np.float32)

        for t in range(signal_length):
            windowed_inputs[t + 1, :] = zero_padded_input[t : t + signal_length]
            timestep_outputs[t + 1] = outputs_np[idx, t]

        causal_inputs_list.append(windowed_inputs)
        causal_outputs_list.append(timestep_outputs)

    causal_inputs_np = np.vstack(causal_inputs_list)  # [N*(T+1), signal_length]
    causal_outputs_np = np.concatenate(causal_outputs_list)  # [N*(T+1)]

    if inputs_is_torch:
        causal_inputs = torch.from_numpy(causal_inputs_np).to(inputs.device)
    else:
        causal_inputs = causal_inputs_np

    if outputs_is_torch:
        causal_outputs = torch.from_numpy(causal_outputs_np).to(outputs.device)
    else:
        causal_outputs = causal_outputs_np

    return causal_inputs, causal_outputs
