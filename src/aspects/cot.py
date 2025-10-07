from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass
class CoTConfig:
    prefix: str = "Let's reason it out step by step."
    temperature: float = 0.6
    min_p: float | None = None


def apply_cot(prompt: str, cfg: CoTConfig) -> Dict[str, object]:
    new_prompt = f"{cfg.prefix}\n{prompt.strip()}"
    kwargs: Dict[str, object] = {"temperature": cfg.temperature}
    if cfg.min_p is not None:
        kwargs["min_p"] = cfg.min_p
    return {"prompt": new_prompt, "kwargs": kwargs}
