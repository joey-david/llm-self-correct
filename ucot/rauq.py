"""Implementation of RAUQ token-level uncertainty scoring."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import torch

from .config import RAUQConfig
from .utils.logging import setup_logging

logger = setup_logging(name=__name__)


@dataclass
class HeadIndices:
    mapping: Dict[int, int]
    layers_used: Optional[List[int]] = None

    @classmethod
    def load(cls, path: Path) -> "HeadIndices":
        """Load stored head indices and layer metadata from `path`."""
        payload = json.loads(path.read_text())
        mapping = {int(k): int(v) for k, v in payload["head_indices"].items()}
        layers_used = payload.get("layers_used")
        return cls(mapping=mapping, layers_used=layers_used)


def select_informative_layers(total_layers: int, fraction: float = 0.33) -> List[int]:
    """Pick a centered span of layers covering the requested fraction of the stack."""
    span = max(1, int(round(total_layers * fraction)))
    start = max(0, (total_layers - span) // 2)
    layers = list(range(start, min(total_layers, start + span)))
    return layers


class RAUQScorer:
    """Stateful RAUQ scorer that maintains per-layer running means."""

    def __init__(
        self,
        alpha: float,
        head_indices: Dict[int, int],
        layers: Sequence[int],
        eps: float = 1e-12,
        device: Optional[torch.device] = None,
    ) -> None:
        """Initialise running RAUQ statistics for the provided layers and heads."""
        self.alpha = alpha
        self.head_indices = head_indices
        self.layers = list(layers)
        self.eps = eps
        self.device = device or torch.device("cpu")
        self.layer_sums = torch.zeros(len(self.layers), device=self.device)
        self.token_counts = torch.zeros(len(self.layers), device=self.device)

    @classmethod
    def from_config(cls, config: RAUQConfig, num_layers: int) -> "RAUQScorer":
        """Build a scorer from configuration defaults and the model's layer count."""
        head_indices = HeadIndices.load(config.head_indices_path)
        layers = head_indices.layers_used or select_informative_layers(num_layers, config.layers_fraction)
        return cls(
            alpha=config.alpha,
            head_indices=head_indices.mapping,
            layers=layers,
            eps=config.eps,
            device=torch.device(config.device if config.device != "auto" else "cpu"),
        )

    def clone(self) -> "RAUQScorer":
        """Return a detached copy of the scorer including accumulated statistics."""
        scorer = RAUQScorer(
            alpha=self.alpha,
            head_indices=self.head_indices,
            layers=self.layers,
            eps=self.eps,
            device=self.device,
        )
        scorer.layer_sums = self.layer_sums.clone()
        scorer.token_counts = self.token_counts.clone()
        return scorer

    def reset(self) -> None:
        """Zero-out the running layer statistics so fresh decoding can begin."""
        self.layer_sums.zero_()
        self.token_counts.zero_()

    def _get_attention_prev_token(self, layer_attn: torch.Tensor, head_idx: int) -> torch.Tensor:
        """Extract the attention weight on the previous token for the selected head."""
        query_len = layer_attn.size(-2)
        key_len = layer_attn.size(-1)
        if key_len < 2:
            return layer_attn.new_tensor(self.eps)
        query_pos = query_len - 1
        prev_key_pos = key_len - 2
        values = layer_attn[:, head_idx, query_pos, prev_key_pos]
        return values.clamp_min(self.eps)

    def _log_prob(self, logits: torch.Tensor, token_id: int | torch.Tensor) -> torch.Tensor:
        """Compute log-probability values for `token_id` from raw logits."""
        log_probs = torch.log_softmax(logits, dim=-1)
        if isinstance(token_id, torch.Tensor):
            index = token_id
        else:
            index = torch.tensor(token_id, device=logits.device)
        if index.ndim == 0:
            index = index.unsqueeze(0)
        if log_probs.ndim == 1:
            return log_probs[index]
        if log_probs.ndim == 2:
            index = index.view(log_probs.size(0), 1)
            return log_probs.gather(-1, index).squeeze(-1)
        raise ValueError(f"Unsupported logits shape: {logits.shape}")

    def score_token(
        self,
        logits: torch.Tensor,
        token_id: int | torch.Tensor,
        attentions: Sequence[torch.Tensor],
        update_state: bool = True,
    ) -> float:
        """Score a single token and optionally fold it into the running statistics."""
        logp = self._log_prob(logits, token_id).mean()

        layer_uncertainties = []
        for idx, layer in enumerate(self.layers):
            head_idx = self.head_indices.get(layer, 0)
            layer_attn = attentions[layer]
            attn_prev = self._get_attention_prev_token(layer_attn, head_idx)
            attn_prev_log = torch.log(attn_prev.clamp_min(self.eps)).mean()
            gamma = self.alpha * logp + (1 - self.alpha) * attn_prev_log
            if update_state:
                self.layer_sums[idx] += gamma
                self.token_counts[idx] += 1
            layer_uncertainties.append(float(-gamma))

        token_score = max(layer_uncertainties)
        return token_score

    def running_rauq(self) -> float:
        """Return the current RAUQ aggregate across layers, or NaN if incomplete."""
        if torch.any(self.token_counts == 0):
            return float("nan")
        means = self.layer_sums / self.token_counts
        return float((-means).max().item())


__all__ = ["RAUQScorer", "HeadIndices", "select_informative_layers"]
