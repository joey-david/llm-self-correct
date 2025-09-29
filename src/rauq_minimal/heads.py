from __future__ import annotations

from typing import Dict, List


class HeadSelector:
    """Selects the most recurrent previous-token head per layer."""

    def select_for_sequence(self, a_prev_all_heads: List[Dict[str, List[float]]]) -> Dict[str, int]:
        if not a_prev_all_heads:
            return {}

        layers = sorted(a_prev_all_heads[0].keys())
        num_tokens = len(a_prev_all_heads)
        selected: Dict[str, int] = {}

        for layer in layers:
            head_counts = len(a_prev_all_heads[0].get(layer, []))
            if head_counts == 0:
                selected[layer] = 0
                continue
            if num_tokens <= 1:
                selected[layer] = 0
                continue
            totals = [0.0] * head_counts
            for token_idx in range(1, num_tokens):
                heads = a_prev_all_heads[token_idx].get(layer, [])
                for h_idx, val in enumerate(heads):
                    totals[h_idx] += float(val)
            best_idx = max(range(head_counts), key=lambda idx: totals[idx])
            selected[layer] = best_idx

        return selected
