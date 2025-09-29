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
        a_prev_all_heads: List[Dict[str, List[float]]],
        selected_heads: Dict[str, int],
    ) -> Tuple[float, List[float]]:
        if not logp_token:
            return 0.0, []

        u_token: List[float] = []
        c_state: Dict[str, float] = {}
        sum_neg_log: Dict[str, float] = {}
        layer_order = sorted(
            selected_heads.keys(),
            key=lambda k: (0, int(k[1:])) if k.startswith("l") and k[1:].isdigit() else (1, k),
        )

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
        else:
            num_tokens = len(logp_token)
            u_final = max(total / num_tokens for total in sum_neg_log.values())

        return u_final, u_token
