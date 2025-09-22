"""Utilities for loading Hugging Face causal language models with attention support."""
from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Optional

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
    attn_implementation: Optional[str] = "eager",
) -> LoadedModel:
    """Load a model/tokenizer pair with sane defaults for experimentation."""

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name or model_name, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    preferred_dtype = torch_dtype or torch.float16

    model_kwargs = {
        "device_map": "auto" if device == "auto" else None,
        "trust_remote_code": trust_remote_code,
    }

    supports_dtype = "dtype" in inspect.signature(AutoModelForCausalLM.from_pretrained).parameters
    if supports_dtype:
        model_kwargs["dtype"] = preferred_dtype
    else:
        model_kwargs["torch_dtype"] = preferred_dtype
    if attn_implementation is not None:
        model_kwargs["attn_implementation"] = attn_implementation

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **model_kwargs,
        )
    except TypeError as error:
        if attn_implementation is not None and "attn_implementation" in str(error):
            logger.warning(
                "attn_implementation=%s not supported by %s; falling back to default loader",
                attn_implementation,
                model_name,
            )
            model_kwargs.pop("attn_implementation", None)
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                **model_kwargs,
            )
        elif supports_dtype and "dtype" in str(error):
            logger.warning("dtype not supported by %s; retrying with torch_dtype", model_name)
            model_kwargs.pop("dtype", None)
            model_kwargs["torch_dtype"] = preferred_dtype
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                **model_kwargs,
            )
        else:
            raise

    if device != "auto":
        model.to(device)
    model.eval()

    if attn_implementation is not None:
        _apply_attn_implementation(model, attn_implementation)

    logger.info("Loaded model %s with device=%s", model_name, device)
    return LoadedModel(model=model, tokenizer=tokenizer)


def _apply_attn_implementation(model: PreTrainedModel, attn_implementation: str) -> None:
    """Force the model to expose attention tensors when requested."""

    applied = False

    setter = getattr(model, "set_attn_implementation", None)
    if callable(setter):
        try:
            setter(attn_implementation)
            applied = True
        except Exception as exc:  # pragma: no cover - depends on model internals
            logger.warning("Unable to set_attn_implementation(%s): %s", attn_implementation, exc)

    config = getattr(model, "config", None)
    if config is not None:
        if hasattr(config, "attn_implementation"):
            setattr(config, "attn_implementation", attn_implementation)
            applied = True
        attn_config = getattr(config, "attn_config", None)
        if isinstance(attn_config, dict):
            attn_config["implementation"] = attn_implementation
            applied = True
        generation_config = getattr(model, "generation_config", None)
        if generation_config is not None and hasattr(generation_config, "attn_implementation"):
            setattr(generation_config, "attn_implementation", attn_implementation)
            applied = True

    for module in model.modules():
        if hasattr(module, "attn_implementation"):
            try:
                setattr(module, "attn_implementation", attn_implementation)
                applied = True
            except Exception as exc:  # pragma: no cover - depends on model internals
                logger.debug(
                    "Unable to set attn_implementation on %s: %s",
                    module.__class__.__name__,
                    exc,
                )
        sub_config = getattr(module, "config", None)
        if sub_config is not None and hasattr(sub_config, "attn_implementation"):
            setattr(sub_config, "attn_implementation", attn_implementation)
            applied = True

    if not applied:
        logger.debug("Model does not expose attn_implementation hooks; proceeding with default settings")


def shift_tokens_right(input_ids: torch.LongTensor, pad_token_id: int) -> torch.LongTensor:
    """Shift tokens right to build decoder input ids as HF does internally."""

    shifted = input_ids.new_zeros(input_ids.shape)
    shifted[..., 1:] = input_ids[..., :-1]
    shifted[..., 0] = pad_token_id
    shifted.masked_fill_(shifted == -100, pad_token_id)
    return shifted
