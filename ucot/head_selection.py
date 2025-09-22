"""Offline head selection for RAUQ."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import torch
from torch.utils.data import DataLoader
from transformers import PreTrainedModel, PreTrainedTokenizer

from .config import HeadSelectionConfig
from .data.utils import batched, load_prompt_completion_pairs
from .utils.logging import setup_logging
from .utils.model import load_model

logger = setup_logging(name=__name__)


@dataclass
class HeadSelectionResult:
    head_indices: Dict[int, int]
    layers_used: List[int]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"head_indices": self.head_indices, "layers_used": self.layers_used}
        path.write_text(json.dumps(payload, indent=2))
        logger.info("Saved head selection to %s", path)

    @classmethod
    def load(cls, path: Path) -> "HeadSelectionResult":
        payload = json.loads(path.read_text())
        return cls(head_indices={int(k): int(v) for k, v in payload["head_indices"].items()}, layers_used=payload["layers_used"])


def _select_informative_layers(num_layers: int, fraction: float = 0.33) -> List[int]:
    span = max(1, int(round(num_layers * fraction)))
    start = (num_layers - span) // 2
    layers = list(range(start, start + span))
    return layers


def _accumulate_attention_statistics(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    samples: Iterable[Tuple[str, str]],
    device: torch.device,
    max_examples: int | None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    num_layers = model.config.num_hidden_layers
    num_heads = model.config.num_attention_heads
    attn_sum = torch.zeros(num_layers, num_heads, device=device)
    token_count = torch.zeros(num_layers, device=device)

    for idx, (prompt, completion) in enumerate(samples):
        if max_examples is not None and idx >= max_examples:
            break
        encoded_prompt = tokenizer(prompt, add_special_tokens=False)
        prompt_ids = encoded_prompt["input_ids"]
        encoded_completion = tokenizer(completion or "", add_special_tokens=False)
        completion_ids = encoded_completion["input_ids"]
        if len(completion_ids) == 0:
            # Fallback: generate one step to get attention stats
            # Teacher forcing requires at least one token; skip empty completions to avoid degenerate entries.
            continue
        input_ids = tokenizer.build_inputs_with_special_tokens(prompt_ids + completion_ids)
        if len(input_ids) < 2:
            continue
        input_tensor = torch.tensor(input_ids, device=device).unsqueeze(0)
        attn_mask = torch.ones_like(input_tensor, device=device)

        with torch.no_grad():
            outputs = model(
                input_ids=input_tensor,
                attention_mask=attn_mask,
                output_attentions=True,
                use_cache=False,
            )
        attentions = outputs.attentions  # tuple[num_layers] => (batch, heads, seq, seq)
        prompt_len = len(tokenizer.build_inputs_with_special_tokens(prompt_ids))
        seq_len = input_tensor.size(1)
        if prompt_len >= seq_len:
            continue
        token_indices = torch.arange(prompt_len, seq_len, device=device)
        prev_indices = token_indices - 1
        for layer_idx, layer_attn in enumerate(attentions):
            # layer_attn: (batch=1, heads, seq, seq)
            head_slice = layer_attn[0, :, token_indices, prev_indices]
            # head_slice shape: (heads, tokens)
            attn_sum[layer_idx] += head_slice.sum(dim=1)
            token_count[layer_idx] += head_slice.size(1)

    return attn_sum, token_count


def select_uncertainty_heads(config: HeadSelectionConfig) -> HeadSelectionResult:
    logger.info("Selecting RAUQ heads for model %s", config.model_name)
    loaded = load_model(
        model_name=config.model_name,
        tokenizer_name=config.tokenizer_name,
        device=config.device,
        torch_dtype=torch.float16,
    )
    model = loaded.model
    tokenizer = loaded.tokenizer

    samples = load_prompt_completion_pairs(config.calibration_paths)
    if not samples:
        raise ValueError("No calibration samples provided")

    device = torch.device(config.device if config.device != "auto" else model.device)
    attn_sum, token_count = _accumulate_attention_statistics(
        model=model,
        tokenizer=tokenizer,
        samples=samples,
        device=device,
        max_examples=config.num_examples,
    )

    if torch.any(token_count == 0):
        logger.warning("Some layers received zero calibration tokens; defaulting to head 0")
        token_count = token_count.masked_fill(token_count == 0, 1.0)

    mean_attn = attn_sum / token_count.unsqueeze(1)
    head_indices = {
        layer_idx: int(torch.argmax(mean_attn[layer_idx]).item())
        for layer_idx in range(mean_attn.size(0))
    }

    layers_used = _select_informative_layers(model.config.num_hidden_layers, config.layers_fraction)
    result = HeadSelectionResult(head_indices=head_indices, layers_used=layers_used)
    result.save(config.output_path)
    return result


__all__ = ["HeadSelectionResult", "select_uncertainty_heads"]
