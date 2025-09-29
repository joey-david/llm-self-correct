from __future__ import annotations

import math
from typing import Dict, List, Tuple


class RAUQ:
    """Compute RAUQ uncertainty metrics for a decoded sequence."""

    def __init__(self, alpha: float = 0.3) -> None:
        self.alpha = alpha

    def compute(
        self,
        logp_token: List[float],
        a_prev_selected: List[Dict[str, float]],
    ) -> Tuple[Dict[str, float], float, List[float]]:
        if not logp_token:
            return {}, 0.0, []

        num_tokens = len(logp_token)
        layers = sorted(a_prev_selected[0].keys()) if a_prev_selected else []

        c_per_layer: Dict[str, List[float]] = {layer: [] for layer in layers}
        u_per_layer: Dict[str, float] = {}
        u_token: List[float] = [0.0] * num_tokens

        for i, logp in enumerate(logp_token):
            prob = math.exp(logp)
            for layer in layers:
                if i == 0:
                    c_val = max(min(prob, 1.0), 1e-12)
                else:
                    prev_c = c_per_layer[layer][i - 1]
                    attn = float(a_prev_selected[i].get(layer, 0.0))
                    c_val = self.alpha * prob + (1.0 - self.alpha) * attn * prev_c
                    c_val = max(min(c_val, 1.0), 1e-12)
                c_per_layer[layer].append(c_val)

        for layer in layers:
            c_values = c_per_layer[layer]
            if not c_values:
                u_per_layer[layer] = 0.0
                continue
            mean_log = sum(math.log(max(val, 1e-12)) for val in c_values) / num_tokens
            u_per_layer[layer] = -mean_log

        for idx in range(num_tokens):
            per_layer_u = [
                -math.log(max(c_per_layer[layer][idx], 1e-12)) for layer in layers
            ]
            u_token[idx] = max(per_layer_u) if per_layer_u else 0.0

        u_final = max(u_per_layer.values()) if u_per_layer else 0.0
        return u_per_layer, u_final, u_token
