from __future__ import annotations

import inspect
from typing import Any, Optional, Tuple

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


class ModelAdapter:
    """Thin wrapper around a Hugging Face causal LM for token-wise decoding."""

    def __init__(
        self,
        model_name: str,
        device: Optional[str] = "auto",
        dtype: str | torch.dtype = "auto",
        attn_implementation: Optional[str] = "eager",
        output_attentions: Optional[bool] = True,
        use_chat_template: Optional[bool] = None,
        trust_remote_code: Optional[bool] = None,
        debug_decode: bool = False,
    ) -> None:
        torch.set_default_dtype(torch.float32)
        self.device = self._resolve_device(device)
        self.dtype = self._resolve_dtype(dtype)

        self.attn_implementation = attn_implementation or None
        self.output_attentions = True if output_attentions is None else bool(output_attentions)
        self.debug_decode = bool(debug_decode)
        self._printed_attn_shapes = False

        model_name_lower = model_name.lower()
        if trust_remote_code is None:
            self.trust_remote_code = "qwen" in model_name_lower
        else:
            self.trust_remote_code = bool(trust_remote_code)

        tokenizer_kwargs: dict[str, Any] = {}
        if self.trust_remote_code:
            tokenizer_kwargs["trust_remote_code"] = True

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, **tokenizer_kwargs)
        if self.tokenizer.pad_token is None:
            if self.tokenizer.eos_token is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            else:
                self.tokenizer.add_special_tokens({"pad_token": "<|pad|>"})

        chat_callable = getattr(self.tokenizer, "apply_chat_template", None)
        self._chat_template = chat_callable if callable(chat_callable) else None
        if use_chat_template is None:
            self.use_chat_template = bool(self._chat_template and getattr(self.tokenizer, "chat_template", None))
        else:
            self.use_chat_template = bool(use_chat_template)

        self._chat_template_supports_enable_thinking = False
        if self._chat_template is not None:
            try:
                signature = inspect.signature(self._chat_template)
                self._chat_template_supports_enable_thinking = "enable_thinking" in signature.parameters
            except (TypeError, ValueError):
                self._chat_template_supports_enable_thinking = False

        config_kwargs: dict[str, Any] = {}
        if self.trust_remote_code:
            config_kwargs["trust_remote_code"] = True
        config = AutoConfig.from_pretrained(model_name, **config_kwargs)
        if self.attn_implementation:
            setattr(config, "attn_implementation", self.attn_implementation)
            setattr(config, "_attn_implementation", self.attn_implementation)
        config.output_attentions = self.output_attentions
        config.use_cache = True
        config.return_dict = True

        model_kwargs: dict[str, Any] = {"device_map": None, "config": config}
        if self.trust_remote_code:
            model_kwargs["trust_remote_code"] = True
        if isinstance(self.dtype, torch.dtype):
            model_kwargs["torch_dtype"] = self.dtype
        if self.attn_implementation:
            model_kwargs["attn_implementation"] = self.attn_implementation
        if self.output_attentions is not None:
            model_kwargs["output_attentions"] = self.output_attentions

        try:
            self.model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        except TypeError:
            model_kwargs.pop("attn_implementation", None)
            try:
                self.model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
            except TypeError:
                model_kwargs.pop("output_attentions", None)
                self.model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

        if hasattr(self.model, "resize_token_embeddings"):
            self.model.resize_token_embeddings(len(self.tokenizer))

        self.model.to(self.device)
        self.model.eval()
        self.config = self.model.config

        if self.attn_implementation:
            if hasattr(self.model.config, "attn_implementation"):
                self.model.config.attn_implementation = self.attn_implementation
            if hasattr(self.model.config, "_attn_implementation"):
                self.model.config._attn_implementation = self.attn_implementation
        self.model.config.output_attentions = self.output_attentions
        self.model.config.use_cache = True
        self.model.config.return_dict = True

        if hasattr(self.model, "generation_config") and self.model.generation_config is not None:
            if self.attn_implementation:
                setattr(self.model.generation_config, "attn_implementation", self.attn_implementation)
                if hasattr(self.model.generation_config, "_attn_implementation"):
                    setattr(self.model.generation_config, "_attn_implementation", self.attn_implementation)
            self.model.generation_config.output_attentions = self.output_attentions
            self.model.generation_config.use_cache = True
            if hasattr(self.model.generation_config, "return_dict_in_generate"):
                self.model.generation_config.return_dict_in_generate = True

    @property
    def eos_token_ids(self) -> Tuple[int, ...]:
        eos = self.tokenizer.eos_token_id
        if eos is None:
            return tuple()
        if isinstance(eos, int):
            return (eos,)
        return tuple(eos)

    def encode(self, prompt: str) -> torch.LongTensor:
        if self.use_chat_template and self._chat_template is not None:
            chat_kwargs: dict[str, Any] = {
                "add_generation_prompt": True,
                "tokenize": True,
                "return_tensors": "pt",
            }
            if self._chat_template_supports_enable_thinking:
                chat_kwargs["enable_thinking"] = False
            encoded = self._chat_template(
                [{"role": "user", "content": prompt}],
                **chat_kwargs,
            )
            input_ids = encoded["input_ids"]
        else:
            encoded = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
            input_ids = encoded["input_ids"]
        return input_ids.to(self.device)

    def step(
        self,
        input_ids: torch.LongTensor,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor, ...], ...]] = None,
        attention_mask: Optional[torch.LongTensor] = None,
    ) -> Tuple[int, float, Tuple[torch.Tensor, ...], Tuple[Tuple[torch.Tensor, ...], ...]]:
        """Run a single forward pass and return greedy token + log-prob and attentions."""
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=self.device)

        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)

        with torch.no_grad():
            forward_kwargs: dict[str, Any] = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "past_key_values": past_key_values,
                "use_cache": True,
                "output_attentions": self.output_attentions,
            }
            if self.attn_implementation:
                forward_kwargs["attn_implementation"] = self.attn_implementation
            try:
                outputs = self.model(**forward_kwargs)
            except TypeError:
                forward_kwargs.pop("attn_implementation", None)
                outputs = self.model(**forward_kwargs)

        logits = outputs.logits[:, -1, :]
        log_probs = torch.log_softmax(logits, dim=-1)
        next_token = torch.argmax(log_probs, dim=-1)
        next_logp = log_probs.gather(-1, next_token.unsqueeze(-1)).squeeze(-1)

        if self.debug_decode:
            top_k = min(5, log_probs.shape[-1])
            top_vals, top_idx = torch.topk(log_probs, k=top_k, dim=-1)

            def _format_token(token_id: int) -> str:
                text = self.tokenizer.decode([token_id], skip_special_tokens=False)
                if not text:
                    pieces = self.tokenizer.convert_ids_to_tokens([token_id])
                    text = pieces[0] if pieces else str(token_id)
                return text.replace("\n", "\\n")

            next_token_id = int(next_token.item())
            decoded_next = _format_token(next_token_id)
            top_formatted = [
                f"{_format_token(idx)}:{score:.3f}"
                for idx, score in zip(top_idx[0].tolist(), top_vals[0].tolist())
            ]
            print(
                f"[ModelAdapter] token={decoded_next!r} (id={next_token_id}) logp={float(next_logp):.3f}; "
                f"topk={top_formatted}"
            )
            if not self._printed_attn_shapes:
                attn = outputs.attentions
                if attn:
                    print(
                        f"[ModelAdapter] attention[0] shape={tuple(attn[0].shape)} "
                        f"layers={len(attn)}"
                    )
                else:
                    print("[ModelAdapter] attentions unavailable (None returned)")
                self._printed_attn_shapes = True

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
