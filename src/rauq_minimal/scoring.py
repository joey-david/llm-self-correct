from __future__ import annotations

import re
from typing import Dict, Iterable, List, Optional


_PUNCT_RE = re.compile(r"[\W_]+", re.UNICODE)
_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")


class AnswerScorer:
    """Utility helpers for normalizing predictions and scoring."""

    def normalize(self, text: str) -> str:
        if text is None:
            return ""
        text = text.strip().lower()
        text = _PUNCT_RE.sub(" ", text)
        return re.sub(r"\s+", " ", text).strip()

    def pick_choice(self, pred_text: str, options: Iterable[str]) -> str:
        options_list = list(options or [])
        if not options_list:
            return pred_text.strip()

        pred_norm = self.normalize(pred_text)
        # Try to detect leading letter selection (A/B/...)
        first_token = pred_norm.split(" ", 1)[0] if pred_norm else ""
        if first_token:
            for idx, opt in enumerate(options_list):
                label = chr(ord("a") + idx)
                if first_token == label:
                    return opt

        option_norms = [self.normalize(opt) for opt in options_list]
        if pred_norm in option_norms:
            return options_list[option_norms.index(pred_norm)]

        pred_tokens = set(pred_norm.split())
        best_idx = 0
        best_overlap = -1
        for idx, opt_norm in enumerate(option_norms):
            opt_tokens = set(opt_norm.split())
            if not opt_tokens:
                continue
            overlap = len(pred_tokens & opt_tokens)
            if overlap > best_overlap:
                best_overlap = overlap
                best_idx = idx
        return options_list[best_idx]

    def extract_numeric(self, pred_text: str) -> Optional[str]:
        if pred_text is None:
            return None
        matches = _NUM_RE.findall(pred_text)
        if not matches:
            return None
        return matches[-1]

    def score(self, record: Dict, pred_eval: str) -> bool:
        answers_norm = record.get("answers_normalized") or []
        if not answers_norm:
            return False
        pred_norm = self.normalize(pred_eval)
        return pred_norm in {a.lower() for a in answers_norm}
