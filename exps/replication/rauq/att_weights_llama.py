#!/usr/bin/env python3
"""Visualise Llama 3.1 8B head-wise attention to the immediately preceding token."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from dotenv import load_dotenv
from huggingface_hub import login as hf_login

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.rauq_minimal.model import ModelAdapter

PROMPT_DEFAULT = "What is King Henry holding in the Portrait of Henry VII?"
_LAYER_PREFIX = "l"
_ENV_PATH = _PROJECT_ROOT / ".env"


def ensure_hf_login() -> None:
    if _ENV_PATH.exists():
        load_dotenv(_ENV_PATH)
    token = os.getenv("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN is not configured in the project .env file.")
    hf_login(token=token, add_to_git_credential=False)


def load_config(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Config at {path} must be a mapping, found {type(data)!r}")
    return data


def should_stop(token_id: int, eos_token_ids: Tuple[int, ...]) -> bool:
    return bool(eos_token_ids) and token_id in eos_token_ids


def format_token(adapter: ModelAdapter, token_id: int) -> str:
    text = adapter.tokenizer.decode([token_id], skip_special_tokens=False)
    if not text:
        pieces = adapter.tokenizer.convert_ids_to_tokens([token_id])
        text = pieces[0] if pieces else str(token_id)
    return text.replace("\n", "\\n")


def make_cumulative_mask(
    past_key_values: Optional[Tuple[Tuple[torch.Tensor, ...], ...]],
    input_len: int,
    device: torch.device,
) -> torch.LongTensor:
    if past_key_values is None:
        return torch.ones((1, input_len), dtype=torch.long, device=device)

    past = past_key_values.past_key_values if hasattr(past_key_values, "past_key_values") else past_key_values
    if not past:
        return torch.ones((1, input_len), dtype=torch.long, device=device)

    first_entry = past[0]
    candidate: Optional[torch.Tensor] = None
    if isinstance(first_entry, (tuple, list)):
        for item in first_entry:
            if torch.is_tensor(item) and item.dim() >= 3:
                candidate = item
                break
    elif torch.is_tensor(first_entry):
        candidate = first_entry

    if candidate is None:
        raise RuntimeError("Unable to infer length of past_key_values for attention mask construction")

    batch = candidate.shape[0]
    past_len = candidate.shape[-2]
    mask_device = candidate.device if candidate.device.type != "meta" else device
    total_len = past_len + input_len
    return torch.ones((batch, total_len), dtype=torch.long, device=mask_device)


def extract_prev_attention(attentions: Optional[Tuple[torch.Tensor, ...]]) -> Dict[str, List[float]]:
    if attentions is None:
        return {}
    attn_per_layer: Dict[str, List[float]] = {}
    for layer_idx, attn in enumerate(attentions):
        layer_attn = attn[0]  # (heads, query_len, key_len)
        heads, q_len, k_len = layer_attn.shape
        if heads == 0:
            attn_per_layer[f"{_LAYER_PREFIX}{layer_idx}"] = []
            continue
        if k_len < 2:
            values = [0.0] * heads
        else:
            prev_idx = k_len - 2
            last_query = q_len - 1
            values = layer_attn[:, last_query, prev_idx].detach().cpu().tolist()
        attn_per_layer[f"{_LAYER_PREFIX}{layer_idx}"] = [float(v) for v in values]
    return attn_per_layer


def zero_attention_template(adapter: ModelAdapter) -> Dict[str, List[float]]:
    num_layers = int(getattr(adapter.config, "num_hidden_layers", 0) or 0)
    num_heads = int(getattr(adapter.config, "num_attention_heads", 0) or 0)
    return {f"{_LAYER_PREFIX}{layer}": [0.0] * num_heads for layer in range(num_layers)}


@torch.inference_mode()
def collect_attention(
    adapter: ModelAdapter,
    prompt: str,
    max_new_tokens: int,
) -> Tuple[List[str], List[int], List[Dict[str, List[float]]], str]:
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive to collect attentions")

    prompt_ids = adapter.encode(prompt)
    prompt_mask = torch.ones_like(prompt_ids, dtype=torch.long, device=adapter.device)

    gen_token_ids: List[int] = []
    token_texts: List[str] = []
    a_prev_all_heads: List[Dict[str, List[float]]] = []

    next_token_id, _, _, past = adapter.step(prompt_ids, past_key_values=None, attention_mask=prompt_mask)
    gen_token_ids.append(next_token_id)
    token_texts.append(format_token(adapter, next_token_id))
    a_prev_all_heads.append({})

    current_input_ids = torch.tensor([[next_token_id]], dtype=torch.long, device=adapter.device)
    token_mask = make_cumulative_mask(past, current_input_ids.shape[1], adapter.device)
    stop = should_stop(next_token_id, adapter.eos_token_ids)

    while len(gen_token_ids) < max_new_tokens and not stop:
        next_token_id, _, attentions, past = adapter.step(
            current_input_ids,
            past_key_values=past,
            attention_mask=token_mask,
        )
        a_prev_all_heads[-1] = extract_prev_attention(attentions)

        gen_token_ids.append(next_token_id)
        token_texts.append(format_token(adapter, next_token_id))
        a_prev_all_heads.append({})

        current_input_ids = torch.tensor([[next_token_id]], dtype=torch.long, device=adapter.device)
        token_mask = make_cumulative_mask(past, current_input_ids.shape[1], adapter.device)
        stop = should_stop(next_token_id, adapter.eos_token_ids)

    if gen_token_ids and a_prev_all_heads[-1] == {}:
        _, _, attentions, _ = adapter.step(
            current_input_ids,
            past_key_values=past,
            attention_mask=make_cumulative_mask(past, current_input_ids.shape[1], adapter.device),
        )
        a_prev_all_heads[-1] = extract_prev_attention(attentions)

    if gen_token_ids:
        template = a_prev_all_heads[0] or zero_attention_template(adapter)
        zero_dict = {layer: [0.0] * len(values) for layer, values in template.items()}
        a_prev_all_heads[0] = zero_dict

    generated_text = adapter.tokenizer.decode(gen_token_ids, skip_special_tokens=True)
    return token_texts, gen_token_ids, a_prev_all_heads, generated_text


def attention_matrix_for_layer(
    a_prev_all_heads: List[Dict[str, List[float]]],
    layer_key: str,
) -> np.ndarray:
    num_tokens = len(a_prev_all_heads)
    num_heads = 0
    for token_heads in a_prev_all_heads:
        heads = token_heads.get(layer_key)
        if heads:
            num_heads = max(num_heads, len(heads))
    if num_heads == 0:
        raise ValueError(f"No attention heads found for layer {layer_key}")

    matrix = np.zeros((num_tokens, num_heads), dtype=np.float32)
    for token_idx, token_heads in enumerate(a_prev_all_heads):
        heads = token_heads.get(layer_key)
        if not heads:
            continue
        upto = min(len(heads), num_heads)
        matrix[token_idx, :upto] = np.asarray(heads[:upto], dtype=np.float32)
    return matrix


def render_heatmap(
    attn_matrix: np.ndarray,
    token_texts: List[str],
    layer_label: str,
    output_path: Path,
    show: bool = False,
) -> None:
    num_tokens, num_heads = attn_matrix.shape
    fig_height = max(4.0, 0.35 * num_tokens)
    fig, ax = plt.subplots(figsize=(0.4 * num_heads + 5.0, fig_height))
    im = ax.imshow(attn_matrix, aspect="auto", cmap="coolwarm", vmin=0.0, vmax=0.7)

    ax.set_xlabel("Attention head")
    ax.set_ylabel("Generated token")
    ax.set_xticks(np.arange(num_heads))
    ax.set_xticklabels([str(idx + 1) for idx in range(num_heads)], rotation=45, ha="right")
    ax.set_yticks(np.arange(num_tokens))
    ax.set_yticklabels(token_texts)
    ax.set_title(f"Layer {layer_label} attention to previous token")

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Attention weight", rotation=90)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    if show:
        plt.show()
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute head-wise previous-token attention weights for Llama 3.1 8B.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to YAML config specifying model and decoding parameters.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=PROMPT_DEFAULT,
        help="Prompt to decode. Defaults to the Henry VII question.",
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=29,
        help="1-based transformer layer index to visualise.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/artifacts/llama31_layer29_attention.png"),
        help="Path to write the attention heatmap figure.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
        help="Optional override for max generated tokens.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the figure interactively after saving.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_hf_login()
    config = load_config(args.config)

    model_name = config.get("model")
    if not model_name:
        raise ValueError("Config must provide a 'model' entry (e.g., meta-llama/Meta-Llama-3.1-8B-Instruct)")

    max_new_tokens = (
        args.max_new_tokens
        if args.max_new_tokens is not None
        else int(config.get("max_new") or config.get("max_new_tokens") or 64)
    )

    adapter = ModelAdapter(
        model_name=model_name,
        device=config.get("device", "auto"),
        dtype=config.get("dtype", "auto"),
        attn_implementation=config.get("attn_implementation", "eager"),
        output_attentions=True,
        use_chat_template=bool(config.get("use_chat_template", False)),
        trust_remote_code=config.get("trust_remote_code"),
        debug_decode=bool(config.get("debug_decode", False)),
    )

    token_texts, _, a_prev_all_heads, generated_text = collect_attention(
        adapter,
        args.prompt,
        max_new_tokens=max_new_tokens,
    )

    layer_index_zero_based = args.layer - 1
    if layer_index_zero_based < 0:
        raise ValueError("Layer index must be >= 1")
    layer_key = f"{_LAYER_PREFIX}{layer_index_zero_based}"

    attn_matrix = attention_matrix_for_layer(a_prev_all_heads, layer_key)
    render_heatmap(attn_matrix, token_texts, str(args.layer), args.output, show=args.show)

    layer_means = attn_matrix.mean(axis=0)
    dominant_head = int(layer_means.argmax()) + 1

    print(f"Prompt: {args.prompt}")
    print(f"Model: {model_name}")
    print(f"Generated text: {generated_text}")
    print(f"Saved attention heatmap to {args.output}")
    print(f"Layer {args.layer} dominant head (mean attention): Head {dominant_head}")
    print("Per-token previous-token attention (averaged across heads):")
    row_means = attn_matrix.mean(axis=1)
    for idx, (token_text, score) in enumerate(zip(token_texts, row_means), start=1):
        print(f"  {idx:02d}: {token_text} -> {score:.4f}")


main()
