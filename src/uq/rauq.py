from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence

import torch

from .head_select import select_heads, selected_prev_attention

EPS = 1e-9


@dataclass
class RAUQOutput:
    u: float
    layer_scores: List[float]
    token_spikes: List[float]
    selected_heads: List[int]


def _mid_layers(num_layers: int) -> List[int]:
    if num_layers <= 0:
        raise ValueError("RAUQ requires at least one layer")
    if num_layers <= 2:
        return list(range(num_layers))
    third = max(num_layers // 3, 1)
    start = (num_layers - third) // 2
    return list(range(start, start + third))


def rauq_score(
    probs: Sequence[float] | torch.Tensor,
    attentions: Sequence[torch.Tensor],
    prompt_len: int,
    alpha: float = 0.2,
    layers_subset: Iterable[int] | None = None,
) -> RAUQOutput:
    if isinstance(probs, torch.Tensor):
        probs_tensor = probs.detach().cpu().float()
    else:
        probs_tensor = torch.tensor(list(probs), dtype=torch.float32)
    probs_tensor = torch.clamp(probs_tensor, min=EPS)
    gen_len = probs_tensor.numel()
    if gen_len == 0:
        raise ValueError("probabilities must include generated tokens")
    heads = select_heads(attentions, prompt_len)
    att = selected_prev_attention(attentions, heads, prompt_len).cpu().float()
    att = torch.clamp(att, min=0.0, max=1.0)
    confidences = []
    for layer_att in att:
        layer_conf = torch.empty(gen_len, dtype=torch.float32)
        layer_conf[0] = probs_tensor[0]
        for i in range(1, gen_len):
            layer_conf[i] = alpha * probs_tensor[i] + (1 - alpha) * layer_att[i] * layer_conf[i - 1]
            layer_conf[i] = torch.clamp(layer_conf[i], min=EPS, max=1.0)
        confidences.append(layer_conf)
    conf_stack = torch.stack(confidences)
    layer_scores = -torch.mean(torch.log(conf_stack), dim=1)
    subset = list(layers_subset) if layers_subset is not None else _mid_layers(len(attentions))
    subset = [l for l in subset if 0 <= l < len(layer_scores)] or _mid_layers(len(layer_scores))
    u_val = torch.max(layer_scores[subset]).item()
    token_spikes = torch.max(-torch.log(conf_stack), dim=0).values.tolist()
    return RAUQOutput(
        u=u_val,
        layer_scores=layer_scores.tolist(),
        token_spikes=token_spikes,
        selected_heads=heads,
    )
