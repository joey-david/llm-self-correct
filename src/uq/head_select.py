from __future__ import annotations

from typing import List, Sequence

import torch


def _slice_generated(attn: torch.Tensor, prompt_len: int) -> torch.Tensor:
    if prompt_len <= 0:
        raise ValueError("prompt must contain at least one token")
    gen_len = attn.size(1) - prompt_len
    if gen_len <= 0:
        raise ValueError("generated sequence must be non-empty")
    if gen_len == 1:
        return torch.empty(attn.size(0), 0, device=attn.device)
    idx = torch.arange(prompt_len, prompt_len + gen_len, device=attn.device)
    cur = idx[1:]
    prev = cur - 1
    values = attn[:, cur, prev]
    return values


def select_heads(attentions: Sequence[torch.Tensor], prompt_len: int) -> List[int]:
    heads: List[int] = []
    for layer_attn in attentions:
        layer = layer_attn[0] if layer_attn.dim() == 4 else layer_attn
        scores = _slice_generated(layer, prompt_len)
        if scores.numel() == 0:
            heads.append(0)
            continue
        mean_scores = scores.mean(dim=1)
        heads.append(int(torch.argmax(mean_scores).item()) if mean_scores.numel() else 0)
    return heads


def selected_prev_attention(
    attentions: Sequence[torch.Tensor], selected: Sequence[int], prompt_len: int
) -> torch.Tensor:
    per_layer = []
    for layer_attn, head_idx in zip(attentions, selected):
        layer = layer_attn[0] if layer_attn.dim() == 4 else layer_attn
        head_vals = _slice_generated(layer, prompt_len)
        first = layer[head_idx, prompt_len : prompt_len + 1, prompt_len - 1 : prompt_len]
        if head_vals.numel() == 0:
            concat = first.flatten()
        else:
            concat = torch.cat([first.flatten(), head_vals[head_idx]])
        per_layer.append(concat)
    return torch.stack(per_layer)
