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
        """Write the head selection result to `path` as a JSON payload."""
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"head_indices": self.head_indices, "layers_used": self.layers_used}
        path.write_text(json.dumps(payload, indent=2))
        logger.info("Saved head selection to %s", path)

    @classmethod
    def load(cls, path: Path) -> "HeadSelectionResult":
        """Restore a head selection result from a JSON file at `path`."""
        payload = json.loads(path.read_text())
        return cls(head_indices={int(k): int(v) for k, v in payload["head_indices"].items()}, layers_used=payload["layers_used"])


def _select_informative_layers(num_layers: int, fraction: float = 0.33) -> List[int]:
    """Choose a contiguous span of layers centred in the stack for calibration."""
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
    show_progress: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Aggregate per-head attention statistics over the provided prompt/completion pairs."""
    num_layers = model.config.num_hidden_layers
    num_heads = model.config.num_attention_heads
    attn_sum = torch.zeros(num_layers, num_heads, device=device)
    token_count = torch.zeros(num_layers, device=device)

    progress_bar = None
    if show_progress:
        total = None
        if hasattr(samples, "__len__"):
            total = len(samples)
            if max_examples is not None:
                total = min(total, max_examples)
        elif max_examples is not None:
            total = max_examples
        try:
            from tqdm.auto import tqdm

            progress_bar = tqdm(total=total, desc="Calibrating heads", unit="example")
        except ImportError:
            progress_bar = None

    for idx, (prompt, completion) in enumerate(samples):
        if max_examples is not None and idx >= max_examples:
            break
        if progress_bar is not None:
            progress_bar.update(1)
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

    if progress_bar is not None:
        progress_bar.close()

    return attn_sum, token_count


def select_uncertainty_heads(config: HeadSelectionConfig) -> HeadSelectionResult:
    """Calibrate attention heads offline and persist the most informative choices."""
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
        show_progress=config.show_progress,
    )

    if torch.any(token_count == 0):
        logger.warning("Some layers received zero calibration tokens; defaulting to head 0")
        token_count = token_count.masked_fill(token_count == 0, 1.0)

    mean_attn = attn_sum / token_count.unsqueeze(1)

    if config.log_head_stats:
        mean_attn_cpu = mean_attn.detach().float().cpu()
        for layer_idx in range(mean_attn_cpu.size(0)):
            layer_scores = mean_attn_cpu[layer_idx]
            if layer_scores.numel() == 0:
                continue
            sorted_scores, sorted_indices = torch.sort(layer_scores, descending=True)
            top_score = sorted_scores[0].item()
            top_idx = sorted_indices[0].item()
            if sorted_scores.numel() > 1:
                runner_score = sorted_scores[1].item()
                runner_idx = sorted_indices[1].item()
                margin = top_score - runner_score
            else:
                runner_score = float("nan")
                runner_idx = -1
                margin = float("nan")
            scores_repr = ", ".join(f"{score:.6f}" for score in layer_scores.tolist())
            logger.info(
                "Layer %d head means: [%s] | top=%d (%.6f) runner-up=%d (%.6f) margin=%.6f",
                layer_idx,
                scores_repr,
                top_idx,
                top_score,
                runner_idx,
                runner_score,
                margin,
            )

    head_indices = {
        layer_idx: int(torch.argmax(mean_attn[layer_idx]).item())
        for layer_idx in range(mean_attn.size(0))
    }

    layers_used = _select_informative_layers(model.config.num_hidden_layers, config.layers_fraction)
    result = HeadSelectionResult(head_indices=head_indices, layers_used=layers_used)
    result.save(config.output_path)
    return result


__all__ = ["HeadSelectionResult", "select_uncertainty_heads"]
