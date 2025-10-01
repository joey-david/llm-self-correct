from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple


class RAUQ:
    """Compute RAUQ uncertainty metrics for a decoded sequence."""

    _VALID_AGGREGATIONS = {"mean", "sum"}

    def __init__(
        self,
        alpha: float = 0.2,
        layer_band: Optional[Tuple[int, int]] = None,
        aggregate_mode: str = "mean",
    ) -> None:
        self.alpha = alpha
        if layer_band is not None:
            low, high = layer_band
            if not isinstance(low, int) or not isinstance(high, int):
                raise TypeError("layer_band must be a tuple of integers (low, high)")
            if low < 0 or high < low:
                raise ValueError("layer_band must satisfy 0 <= low <= high")
        self.layer_band = layer_band
        aggregate_mode = aggregate_mode.lower().strip()
        if aggregate_mode not in self._VALID_AGGREGATIONS:
            raise ValueError(
                f"unsupported aggregate_mode={aggregate_mode!r}; expected one of {sorted(self._VALID_AGGREGATIONS)}"
            )
        self.aggregate_mode = aggregate_mode

    def compute(
        self,
        logp_token: List[float],
        a_prev_all_heads: List[Dict[str, List[float]]],
        selected_heads: Dict[str, int],
        layer_band: Optional[Tuple[int, int]] = None,
        aggregate_mode: Optional[str] = None,
    ) -> Tuple[float, List[float], Optional[str], Optional[int]]:
        if not logp_token:
            return 0.0, [], None, None

        agg_mode = (aggregate_mode or self.aggregate_mode).lower().strip()
        if agg_mode not in self._VALID_AGGREGATIONS:
            raise ValueError(
                f"unsupported aggregate_mode={agg_mode!r}; expected one of {sorted(self._VALID_AGGREGATIONS)}"
            )

        u_token: List[float] = []
        c_state: Dict[str, float] = {}
        layer_sums: Dict[str, float] = {}
        layer_counts: Dict[str, int] = {}
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
            token_max_u: Optional[float] = None

            if not layer_order:
                token_max_u = -math.log(prob)
                u_token.append(token_max_u)
                continue

            for layer in layer_order:
                head_idx = selected_heads.get(layer, 0)
                heads = token_heads.get(layer, [])
                attn = None
                if head_idx < len(heads) and len(heads) > 0:
                    try:
                        attn = float(heads[head_idx])
                    except (TypeError, ValueError):
                        attn = None
                if attn is None or not math.isfinite(attn):
                    attn = 0.0
                attn = max(min(attn, 1.0), 0.0)

                if idx == 0:
                    c_val = prob
                else:
                    prev_c = c_state.get(layer, 1.0)
                    c_val = self.alpha * prob + (1.0 - self.alpha) * attn * prev_c

                c_val = max(min(c_val, 1.0), 1e-12)
                c_state[layer] = c_val

                neg_log = -math.log(c_val)
                token_max_u = neg_log if token_max_u is None else max(token_max_u, neg_log)

                if idx > 0:
                    # Only aggregate over generated tokens that have a previous generated token,
                    # matching the paper's summation from i=2..N.
                    layer_sums[layer] = layer_sums.get(layer, 0.0) + neg_log
                    layer_counts[layer] = layer_counts.get(layer, 0) + 1

            u_token.append(token_max_u if token_max_u is not None else -math.log(prob))

        if not layer_sums:
            u_final = max(u_token) if u_token else 0.0
            best_layer = None
        else:
            best_layer = None
            best_value = float("-inf")
            for layer, total in layer_sums.items():
                count = layer_counts.get(layer, 0)
                if count <= 0:
                    continue
                layer_u = total if agg_mode == "sum" else total / count
                if layer_u > best_value:
                    best_value = layer_u
                    best_layer = layer
            if best_layer is None:
                u_final = max(u_token) if u_token else 0.0
            else:
                u_final = layer_sums[best_layer] if agg_mode == "sum" else layer_sums[best_layer] / max(
                    1, layer_counts.get(best_layer, 1)
                )

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
