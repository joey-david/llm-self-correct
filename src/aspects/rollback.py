from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass
class RollbackConfig:
    depth: int = 3
    temperature: float = 0.3
    top_p: float = 0.7
    repetition_penalty: float = 1.2


def plan_rollback(tokens: List[int], spike_index: int, cfg: RollbackConfig) -> Dict[str, object]:
    start = max(0, spike_index - cfg.depth)
    truncated = tokens[:start]
    regen_kwargs = {
        "temperature": cfg.temperature,
        "top_p": cfg.top_p,
        "repetition_penalty": cfg.repetition_penalty,
    }
    return {"prefix_tokens": truncated, "regen_kwargs": regen_kwargs}
