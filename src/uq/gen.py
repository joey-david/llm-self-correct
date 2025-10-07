from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .io import enforce_determinism, require_gpu


@dataclass
class GenerationOutput:
    prompt_len: int
    tokens: torch.Tensor
    token_probs: torch.Tensor
    scores: torch.Tensor
    attentions: list[torch.Tensor]
    text: str


class HFGenerator:
    def __init__(
        self,
        model_name: str = "meta-llama/Meta-Llama-3.1-8B",
        force_cpu: bool = False,
        torch_dtype: Optional[torch.dtype] = None,
    ) -> None:
        enforce_determinism()
        require_gpu(force_cpu=force_cpu)
        self.device = torch.device("cpu" if force_cpu else "cuda")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            attn_implementation="eager",
        ).to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def generate(self, prompt: str, max_new_tokens: int = 256) -> GenerationOutput:
        encoded = self.tokenizer(prompt, return_tensors="pt")
        encoded = {k: v.to(self.device) for k, v in encoded.items()}
        prompt_len = encoded["input_ids"].size(-1)
        outputs = self.model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            output_scores=True,
            output_attentions=True,
            return_dict_in_generate=True,
            use_cache=True,
        )
        sequences = outputs.sequences
        new_tokens = sequences[:, prompt_len:]
        scores = torch.stack(outputs.scores, dim=1) if outputs.scores else torch.empty(1, 0, self.model.config.vocab_size)
        probs = torch.softmax(scores, dim=-1)
        token_probs = torch.gather(probs, dim=-1, index=new_tokens.unsqueeze(-1)).squeeze(-1)
        # stitch attentions (list over steps -> per-layer tensors)
        num_layers = len(outputs.attentions[0]) if outputs.attentions else 0
        stitched = []
        for layer_idx in range(num_layers):
            num_heads = outputs.attentions[0][layer_idx].size(1)
            total_len = prompt_len + new_tokens.size(1)
            buf = torch.zeros(1, num_heads, total_len, total_len, device="cpu")
            stitched.append(buf)
        for step, per_layer in enumerate(outputs.attentions):
            cur_len = prompt_len + step + 1
            for layer_idx, layer_attn in enumerate(per_layer):
                buf = stitched[layer_idx]
                buf[:, :, :cur_len, :cur_len] = layer_attn.detach().cpu()
        decoded = self.tokenizer.batch_decode(sequences[:, prompt_len:], skip_special_tokens=True)
        return GenerationOutput(
            prompt_len=prompt_len,
            tokens=new_tokens.detach().cpu(),
            token_probs=token_probs.detach().cpu()[0],
            scores=scores.detach().cpu()[0],
            attentions=[layer[0] for layer in stitched],
            text=decoded[0],
        )
