from __future__ import annotations

from dataclasses import dataclass
from statistics import median
from typing import Iterable, List


@dataclass
class SpikeConfig:
    tau_abs: float = 5.0
    tau_rel: float = 2.0
    window: int = 4
    refractory: int = 2


def rolling_z(values: Iterable[float], window: int = 5) -> List[float]:
    vals = list(values)
    out: List[float] = []
    for i, v in enumerate(vals):
        start = max(0, i - window)
        slice_vals = vals[start : i + 1]
        mean = sum(slice_vals) / len(slice_vals)
        var = sum((x - mean) ** 2 for x in slice_vals) / len(slice_vals)
        std = var**0.5
        out.append(0.0 if std == 0 else (v - mean) / std)
    return out


def detect_spikes(scores: Iterable[float], cfg: SpikeConfig) -> List[int]:
    vals = list(scores)
    flagged: List[int] = []
    last_spike = -10**6
    history: List[float] = []
    for idx, val in enumerate(vals):
        history.append(val)
        recent = history[max(0, len(history) - cfg.window) :]
        med = median(recent) if recent else 0.0
        is_spike = val > cfg.tau_abs or (val - med) > cfg.tau_rel
        if is_spike and idx - last_spike >= cfg.refractory:
            flagged.append(idx)
            last_spike = idx
    return flagged
