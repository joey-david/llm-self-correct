"""Utilities for handling `past_key_values` caches."""
from __future__ import annotations

from typing import Iterable, Tuple

import torch

PastKeyValues = Tuple[Tuple[torch.Tensor, torch.Tensor], ...]


def slice_past_key_values(past: PastKeyValues, length: int) -> PastKeyValues:
    """Slice cached key/values to a specific sequence length.

    Hugging Face caches use shape (batch, heads, seq, head_dim). We truncate the sequence dimension to `length` across all layers.
    """

    if past is None:
        return past
    sliced = []
    for keys, values in past:
        if keys.size(-2) < length:
            sliced.append((keys, values))
            continue
        sliced.append((keys[..., :length, :], values[..., :length, :]))
    return tuple(sliced)


def detach_past_key_values(past: PastKeyValues) -> PastKeyValues:
    """Detach cached tensors from the computation graph to reduce memory."""

    if past is None:
        return past
    detached = []
    for keys, values in past:
        detached.append((keys.detach(), values.detach()))
    return tuple(detached)


def to_device(past: PastKeyValues, device: torch.device) -> PastKeyValues:
    """Move cached tensors to a specific device."""

    if past is None:
        return past
    moved = []
    for keys, values in past:
        moved.append((keys.to(device=device), values.to(device=device)))
    return tuple(moved)


__all__ = ["slice_past_key_values", "detach_past_key_values", "to_device", "PastKeyValues"]
