from __future__ import annotations

from typing import Any, Union

import torch
from transformers.file_utils import ModelOutput
from transformers.modeling_outputs import SequenceClassifierOutput


def detach(input: Union[torch.Tensor, dict[Any, torch.Tensor], ModelOutput]):
    """Detaches tensors from Pytorch graph.
    Note: torch.tensor.detach() returns a view of original tensor without autograd history.
    Args:
        input: Pytorch tensor, dict with tensor values, or huggingface transformer ModelOutput
    Returns:
        Dict with keys and tensor values that have been detached from the pytorch graph.
    """
    if isinstance(input, torch.Tensor):
        return input.detach()
    if isinstance(input, dict):
        return {
            k: (v.detach() if isinstance(v, torch.Tensor) else v)
            for k, v in input.items()
        }


def combine_batches(batches: list[Union[dict, SequenceClassifierOutput]]) -> dict:
    "Given a list of batches or outputs, combines them by key to produce single dict."

    # Convert SequenceClassifierOutput to dict
    if isinstance(batches, SequenceClassifierOutput):
        batches = [{key: batch[key] for key in batch.keys()} for batch in batches]

    merged = {}
    for batch in batches:
        for key in batch.keys():
            value = batch[key]
            if value is None:
                continue
            # If value is integer or float, wrap in list
            if isinstance(value, int):
                value = [value]
            if isinstance(value, float):
                value = [value]
            # If value is 0-dimensional tensor, make into 1-dimensional tensor
            if isinstance(value, torch.Tensor) and not bool(value.shape):
                value = value.unsqueeze(dim=0)
            # Accumulate into lists organized by keys
            if key in merged:
                merged[key] += value
            else:
                merged[key] = list(value)

    # If value is tensor, vertical stack them together
    for key in merged.keys():
        if isinstance(merged[key][0], torch.Tensor):
            merged[key] = torch.stack([x for x in merged[key]])
    return merged
