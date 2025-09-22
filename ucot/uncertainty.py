"""Generic uncertainty scorers used by the controller and ablations."""
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Optional, Sequence

import torch

from .rauq import RAUQScorer


class TokenUncertaintyScorer:
    """Base interface for per-token uncertainty scorers."""

    def clone(self) -> "TokenUncertaintyScorer":  # pragma: no cover - interface only
        raise NotImplementedError

    def reset(self) -> None:  # pragma: no cover - interface only
        raise NotImplementedError

    def score_token(
        self,
        logits: torch.Tensor,
        token_id: int | torch.Tensor,
        attentions: Sequence[torch.Tensor] | None = None,
        update_state: bool = True,
    ) -> float:  # pragma: no cover - interface only
        raise NotImplementedError


class EntropyScorer(TokenUncertaintyScorer):
    """Simple token entropy-based uncertainty."""

    def __init__(self) -> None:
        self._history: list[float] = []

    def clone(self) -> "EntropyScorer":
        cloned = EntropyScorer()
        cloned._history = self._history.copy()
        return cloned

    def reset(self) -> None:
        self._history.clear()

    def score_token(self, logits: torch.Tensor, token_id: int | torch.Tensor, **_) -> float:
        log_probs = torch.log_softmax(logits, dim=-1)
        probs = log_probs.exp()
        entropy = -(probs * log_probs).sum().item()
        self._history.append(entropy)
        return entropy


class LogitMarginScorer(TokenUncertaintyScorer):
    """Margin between top-1 and top-2 logits; higher margin -> lower uncertainty."""

    def __init__(self) -> None:
        self._history: list[float] = []

    def clone(self) -> "LogitMarginScorer":
        cloned = LogitMarginScorer()
        cloned._history = self._history.copy()
        return cloned

    def reset(self) -> None:
        self._history.clear()

    def score_token(self, logits: torch.Tensor, token_id: int | torch.Tensor, **_) -> float:
        top2 = torch.topk(logits, k=2, dim=-1).values
        if top2.size(-1) < 2:
            margin = float("inf")
        else:
            margin = (top2[..., 0] - top2[..., 1]).item()
        uncertainty = -margin
        self._history.append(uncertainty)
        return uncertainty


class RAUQScorerWrapper(RAUQScorer):
    """Subclasses RAUQScorer to conform to TokenUncertaintyScorer signature."""

    def clone(self) -> "RAUQScorerWrapper":
        new = RAUQScorerWrapper(
            alpha=self.alpha,
            head_indices=self.head_indices,
            layers=self.layers,
            eps=self.eps,
            device=self.device,
        )
        new.layer_sums = self.layer_sums.clone()
        new.token_counts = self.token_counts.clone()
        return new


__all__ = [
    "TokenUncertaintyScorer",
    "EntropyScorer",
    "LogitMarginScorer",
    "RAUQScorerWrapper",
]
