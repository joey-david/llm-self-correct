from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Dict, Sequence

import torch

from .head_select import select_heads, selected_prev_attention


@dataclass
class BaselineResult:
    name: str
    sequence: float
    token_scores: list[float]
    extras: Dict[str, object] | None = None


def msp_perplexity(probs: Sequence[float] | torch.Tensor) -> BaselineResult:
    probs_tensor = (
        probs.detach().cpu().float() if isinstance(probs, torch.Tensor) else torch.tensor(list(probs), dtype=torch.float32)
    )
    probs_tensor = torch.clamp(probs_tensor, min=1e-9, max=1.0)
    nll = -torch.log(probs_tensor)
    return BaselineResult("msp", float(torch.mean(nll).item()), nll.tolist())


def attention_score(
    attentions: Sequence[torch.Tensor],
    prompt_len: int,
    use_selected_head: bool = True,
) -> BaselineResult:
    if use_selected_head:
        heads = select_heads(attentions, prompt_len)
        att = selected_prev_attention(attentions, heads, prompt_len).cpu()
    else:
        stacked = []
        for layer in attentions:
            layer = layer[0] if layer.dim() == 4 else layer
            stacked.append(layer[:, prompt_len:, prompt_len - 1 : prompt_len].mean(dim=0))
        att = torch.stack(stacked)
        heads = None
    per_token = 1.0 - torch.clamp(att.mean(dim=0), min=0.0, max=1.0)
    score = float(per_token.mean().item())
    return BaselineResult("attention", score, per_token.tolist(), {"selected_heads": heads})


def semantic_entropy(samples: Sequence[torch.Tensor]) -> BaselineResult:
    if not samples:
        raise ValueError("semantic entropy requires at least one sample distribution")
    stack = torch.stack([s.cpu().float() for s in samples])
    stack = stack.clamp_min(1e-9)
    stack = stack / stack.sum(dim=-1, keepdim=True)
    entropy = -torch.sum(stack * torch.log(stack), dim=-1).mean(dim=0)
    return BaselineResult("semantic_entropy", float(entropy.mean().item()), entropy.tolist())


def warn_stub(name: str) -> BaselineResult:
    warnings.warn(f"Baseline {name} not implemented; returning NaNs", UserWarning)
    nan = float("nan")
    return BaselineResult(name, nan, [nan])


def get_baseline(name: str):
    table = {
        "msp": msp_perplexity,
        "perplexity": msp_perplexity,
        "attention": attention_score,
        "semantic_entropy": semantic_entropy,
        "focus": lambda *_args, **_kwargs: warn_stub("focus"),
        "ccp": lambda *_args, **_kwargs: warn_stub("ccp"),
        "sar": lambda *_args, **_kwargs: warn_stub("sar"),
        "luq": lambda *_args, **_kwargs: warn_stub("luq"),
        "eigenscore": lambda *_args, **_kwargs: warn_stub("eigenscore"),
    }
    return table[name]
