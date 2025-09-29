from __future__ import annotations

from typing import Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class ModelAdapter:
    """Thin wrapper around a Hugging Face causal LM for token-wise decoding."""

    def __init__(
        self,
        model_name: str,
        device: Optional[str] = "auto",
        dtype: str | torch.dtype = "auto",
    ) -> None:
        torch.set_default_dtype(torch.float32)
        self.device = self._resolve_device(device)
        self.dtype = self._resolve_dtype(dtype)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            if self.tokenizer.eos_token is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            else:
                self.tokenizer.add_special_tokens({"pad_token": "<|pad|>"})

        model_kwargs = {"device_map": None, "attn_implementation": "eager"}
        if isinstance(self.dtype, torch.dtype):
            model_kwargs["torch_dtype"] = self.dtype

        try:
            self.model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        except TypeError:
            # Older transformer builds might not accept attn_implementation.
            model_kwargs.pop("attn_implementation", None)
            self.model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        if hasattr(self.model, "resize_token_embeddings"):
            self.model.resize_token_embeddings(len(self.tokenizer))

        self.model.to(self.device)
        self.model.eval()
        self.config = self.model.config

        if hasattr(self.model.config, "attn_implementation"):
            self.model.config.attn_implementation = "eager"
        if hasattr(self.model.config, "_attn_implementation"):
            self.model.config._attn_implementation = "eager"

    @property
    def eos_token_ids(self) -> Tuple[int, ...]:
        eos = self.tokenizer.eos_token_id
        if eos is None:
            return tuple()
        if isinstance(eos, int):
            return (eos,)
        return tuple(eos)

    def encode(self, prompt: str) -> torch.LongTensor:
        encoded = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        return encoded["input_ids"].to(self.device)

    def step(
        self,
        input_ids: torch.LongTensor,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor, ...], ...]] = None,
        attention_mask: Optional[torch.LongTensor] = None,
    ) -> Tuple[int, float, Tuple[torch.Tensor, ...], Tuple[Tuple[torch.Tensor, ...], ...]]:
        """Run a single forward pass and return greedy token + log-prob and attentions."""
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=self.device)

        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids.to(self.device),
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
                output_attentions=True,
                attn_implementation="eager",
            )

        logits = outputs.logits[:, -1, :]
        log_probs = torch.log_softmax(logits, dim=-1)
        next_token = torch.argmax(log_probs, dim=-1)
        next_logp = log_probs.gather(-1, next_token.unsqueeze(-1)).squeeze(-1)

        return (
            int(next_token.item()),
            float(next_logp.item()),
            outputs.attentions,
            outputs.past_key_values,
        )

    def _resolve_device(self, device: Optional[str]) -> torch.device:
        if device in (None, "auto"):
            if torch.cuda.is_available():
                return torch.device("cuda")
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")
        return torch.device(device)

    def _resolve_dtype(self, dtype: str | torch.dtype) -> torch.dtype:
        if isinstance(dtype, torch.dtype):
            return dtype
        if dtype == "auto":
            if self.device.type == "cuda":
                return torch.float16
            return torch.float32
        if isinstance(dtype, str):
            resolved = getattr(torch, dtype, None)
            if resolved is None or not isinstance(resolved, torch.dtype):
                raise ValueError(f"Unsupported dtype: {dtype}")
            return resolved
        raise ValueError(f"Unsupported dtype: {dtype}")
