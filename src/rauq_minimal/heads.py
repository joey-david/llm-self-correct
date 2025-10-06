from __future__ import annotations

from typing import Dict, Iterable, List, Optional


class HeadSelector:
    """Selects the most recurrent previous-token head per layer."""

    def __init__(self) -> None:
        self._totals: Dict[str, List[float]] = {}
        self._counts: Dict[str, List[int]] = {}
        self._head_counts: Dict[str, int] = {}
        self._selected: Optional[Dict[str, int]] = None
        self._num_sequences: int = 0

    def reset(self) -> None:
        self._totals.clear()
        self._counts.clear()
        self._head_counts.clear()
        self._selected = None
        self._num_sequences = 0

    def observe_sequence(self, a_prev_all_heads: List[Dict[str, List[float]]]) -> None:
        """Accumulate per-head attention scores from a decoded sequence."""

        if self._selected is not None or not a_prev_all_heads:
            return

        layer_names = set()
        for token_heads in a_prev_all_heads:
            layer_names.update(token_heads.keys())
        layers = sorted(layer_names)
        num_tokens = len(a_prev_all_heads)
        if num_tokens <= 1:
            for layer in layers:
                self._head_counts[layer] = self._layer_head_count(layer, a_prev_all_heads)
            self._num_sequences += 1
            return

        for layer in layers:
            head_counts = self._layer_head_count(layer, a_prev_all_heads)
            self._head_counts[layer] = max(self._head_counts.get(layer, 0), head_counts)
            if head_counts == 0:
                continue

            totals = self._totals.setdefault(layer, [0.0] * head_counts)
            counts = self._counts.setdefault(layer, [0] * head_counts)
            if len(totals) < head_counts:
                totals.extend([0.0] * (head_counts - len(totals)))
            if len(counts) < head_counts:
                counts.extend([0] * (head_counts - len(counts)))

            for token_idx in range(1, num_tokens):
                heads = a_prev_all_heads[token_idx].get(layer, [])
                upto = min(len(heads), head_counts)
                for h_idx in range(upto):
                    totals[h_idx] += float(heads[h_idx])
                    counts[h_idx] += 1

        self._num_sequences += 1

    def fit(self, sequences: Iterable[List[Dict[str, List[float]]]]) -> Dict[str, int]:
        for seq in sequences:
            self.observe_sequence(seq)
        return self.finalize()

    def finalize(self) -> Dict[str, int]:
        if self._selected is not None:
            return dict(self._selected)

        selected: Dict[str, int] = {}
        all_layers = set(self._head_counts.keys()) | set(self._totals.keys())
        for layer in sorted(all_layers):
            head_count = self._head_counts.get(layer, 0)
            totals = self._totals.get(layer, [])
            counts = self._counts.get(layer, [])
            if head_count == 0:
                # Nothing to select for this layer; skip so it cannot influence downstream scoring.
                continue

            if len(totals) < head_count:
                totals = totals + [0.0] * (head_count - len(totals))
            if len(counts) < head_count:
                counts = counts + [0] * (head_count - len(counts))

            # Compute mean attention per head; skip heads never observed
            means: List[float] = []
            for total, count in zip(totals, counts):
                if count <= 0:
                    means.append(float("-inf"))
                else:
                    means.append(total / count)

            if all(m == float("-inf") for m in means):
                # No signal collected for this layer; skip entirely
                continue

            best_idx = max(range(len(means)), key=means.__getitem__)
            selected[layer] = int(best_idx)

        self._selected = selected
        return dict(selected)

    def is_ready(self) -> bool:
        return self._selected is not None

    def get_selected(self) -> Dict[str, int]:
        if self._selected is None:
            raise RuntimeError("HeadSelector has not been finalized. Run finalize() first.")
        return dict(self._selected)

    def select_for_sequence(self, a_prev_all_heads: List[Dict[str, List[float]]]) -> Dict[str, int]:
        """Backward-compatible helper returning the frozen head selection."""

        if self._selected is None:
            raise RuntimeError(
                "HeadSelector.select_for_sequence() is deprecated for per-sequence estimation. "
                "Call observe_sequence()/finalize() during calibration and get_selected() afterwards."
            )
        return dict(self._selected)

    @property
    def num_sequences(self) -> int:
        return self._num_sequences

    @staticmethod
    def _layer_head_count(
        layer: str, a_prev_all_heads: List[Dict[str, List[float]]]
    ) -> int:
        for token_heads in a_prev_all_heads:
            heads = token_heads.get(layer)
            if heads is not None:
                return len(heads)
        return 0
