from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Tuple


@dataclass
class UtilityWeights:
    value: float = 1.0
    cost_cot: float = 0.1
    cost_rb: float = 0.01
    cost_latency: float = 0.001


def compute_utility(
    delta_acc: float,
    cot_used: bool,
    rollback_tokens: int,
    delta_time: float,
    weights: UtilityWeights,
) -> float:
    utility = delta_acc * weights.value
    if cot_used:
        utility -= weights.cost_cot
    utility -= rollback_tokens * weights.cost_rb
    utility -= delta_time * weights.cost_latency
    return utility


def calibrate_configs(
    configs: Iterable[Dict[str, float]],
    stats: Dict[str, Dict[str, float]],
    weights: UtilityWeights,
) -> Tuple[Dict[str, float], float]:
    best_cfg: Dict[str, float] | None = None
    best_utility = float("-inf")
    for cfg in configs:
        name = cfg.get("name", str(cfg))
        data = stats.get(name, {})
        util = compute_utility(
            data.get("delta_acc", 0.0),
            bool(data.get("cot_used", 0)),
            int(data.get("rollback_tokens", 0)),
            data.get("delta_time", 0.0),
            weights,
        )
        if util > best_utility:
            best_cfg = dict(cfg)
            best_utility = util
    if best_cfg is None:
        raise ValueError("No calibration configs provided")
    return best_cfg, best_utility
