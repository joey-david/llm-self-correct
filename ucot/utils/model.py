"""Utilities for loading Hugging Face causal language models with attention support."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer

from .logging import setup_logging

logger = setup_logging(name=__name__)


@dataclass
class LoadedModel:
    model: PreTrainedModel
    tokenizer: PreTrainedTokenizer


def load_model(
    model_name: str,
    tokenizer_name: Optional[str] = None,
    device: str = "cuda",
    torch_dtype: Optional[torch.dtype] = None,
    trust_remote_code: bool = False,
) -> LoadedModel:
    """Load a model/tokenizer pair with sane defaults for experimentation."""

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name or model_name, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto" if device == "auto" else None,
        torch_dtype=torch_dtype or torch.float16,
        trust_remote_code=trust_remote_code,
    )
    if device != "auto":
        model.to(device)
    model.eval()

    logger.info("Loaded model %s with device=%s", model_name, device)
    return LoadedModel(model=model, tokenizer=tokenizer)


def shift_tokens_right(input_ids: torch.LongTensor, pad_token_id: int) -> torch.LongTensor:
    """Shift tokens right to build decoder input ids as HF does internally."""

    shifted = input_ids.new_zeros(input_ids.shape)
    shifted[..., 1:] = input_ids[..., :-1]
    shifted[..., 0] = pad_token_id
    shifted.masked_fill_(shifted == -100, pad_token_id)
    return shifted
