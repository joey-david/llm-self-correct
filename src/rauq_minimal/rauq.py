from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple


class RAUQ:
    """Compute RAUQ uncertainty metrics for a decoded sequence."""

    def __init__(self, alpha: float = 0.2, layer_band: Optional[Tuple[int, int]] = None) -> None:
        self.alpha = alpha
        if layer_band is not None:
            low, high = layer_band
            if not isinstance(low, int) or not isinstance(high, int):
                raise TypeError("layer_band must be a tuple of integers (low, high)")
            if low < 0 or high < low:
                raise ValueError("layer_band must satisfy 0 <= low <= high")
        self.layer_band = layer_band

    def compute(
        self,
        logp_token: List[float],
        a_prev_all_heads: List[Dict[str, List[float]]],
        selected_heads: Dict[str, int],
        layer_band: Optional[Tuple[int, int]] = None,
    ) -> Tuple[float, List[float], Optional[str], Optional[int]]:
        if not logp_token:
            return 0.0, [], None, None

        u_token: List[float] = []
        c_state: Dict[str, float] = {}
        sum_neg_log: Dict[str, float] = {}
        band = layer_band if layer_band is not None else self.layer_band
        layer_order = sorted(
            selected_heads.keys(),
            key=lambda k: (0, int(k[1:])) if k.startswith("l") and k[1:].isdigit() else (1, k),
        )

        if band is not None:
            low, high = band
            filtered_layers = []
            for layer in layer_order:
                idx = self._layer_index(layer)
                if idx is None:
                    continue
                if low <= idx <= high:
                    filtered_layers.append(layer)
            layer_order = filtered_layers

        for idx, logp in enumerate(logp_token):
            prob = max(min(math.exp(logp), 1.0), 1e-12)
            token_heads = a_prev_all_heads[idx] if idx < len(a_prev_all_heads) else {}
            token_max_u = 0.0

            if not layer_order:
                token_max_u = -math.log(prob)
                u_token.append(token_max_u)
                continue

            for layer in layer_order:
                head_idx = selected_heads.get(layer, 0)
                heads = token_heads.get(layer, [])
                attn = float(heads[head_idx]) if head_idx < len(heads) else 0.0

                if idx == 0:
                    c_val = prob
                else:
                    prev_c = c_state.get(layer, prob)
                    c_val = self.alpha * prob + (1.0 - self.alpha) * attn * prev_c

                c_val = max(min(c_val, 1.0), 1e-12)
                c_state[layer] = c_val

                neg_log = -math.log(c_val)
                sum_neg_log[layer] = sum_neg_log.get(layer, 0.0) + neg_log
                token_max_u = max(token_max_u, neg_log)

            u_token.append(token_max_u)

        if not sum_neg_log:
            u_final = max(u_token) if u_token else 0.0
            best_layer = None
        else:
            num_tokens = len(logp_token)
            best_layer = None
            best_value = float("-inf")
            for layer, total in sum_neg_log.items():
                layer_u = total / num_tokens
                if layer_u > best_value:
                    best_value = layer_u
                    best_layer = layer
            u_final = best_value if best_layer is not None else 0.0

        best_head: Optional[int]
        if best_layer is not None:
            best_head = selected_heads.get(best_layer)
        else:
            best_head = None

        return u_final, u_token, best_layer, best_head

    @staticmethod
    def _layer_index(layer_name: str) -> Optional[int]:
        if layer_name.startswith("l") and layer_name[1:].isdigit():
            return int(layer_name[1:])
        return None
