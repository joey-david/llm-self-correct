from __future__ import annotations

import json
import math
import random
from typing import Dict, List, Optional, Set

import numpy as np
import torch
from tqdm.auto import tqdm

from .heads import HeadSelector
from .model import ModelAdapter
from .prompt import PromptBuilder
from .rauq import RAUQ
from .scoring import AnswerScorer


class Runner:
    """Main orchestration for decoding, collecting stats, and writing output."""

    def __init__(
        self,
        model_adapter: ModelAdapter,
        prompt_builder: PromptBuilder,
        scorer: AnswerScorer,
        head_selector: HeadSelector,
        rauq: RAUQ,
        max_new_tokens: int,
        store_all_heads: bool = False,
        dataset_fraction: float = 1.0,
        debug_decode: bool = False,
    ) -> None:
        self.model_adapter = model_adapter
        self.prompt_builder = prompt_builder
        self.scorer = scorer
        self.head_selector = head_selector
        self.rauq = rauq
        self.max_new_tokens = max_new_tokens
        self.store_all_heads = store_all_heads
        if dataset_fraction < 0.0 or dataset_fraction > 1.0:
            raise ValueError("dataset_fraction must be between 0 and 1")
        self.dataset_fraction = dataset_fraction
        self.debug_decode = debug_decode

    def run(self, input_path: str, output_path: str) -> None:
        selected_indices: Optional[Set[int]] = None
        if 0.0 < self.dataset_fraction < 1.0:
            selected_indices = self._select_record_indices(input_path)
        elif self.dataset_fraction == 0.0:
            selected_indices = set()

        if selected_indices is None:
            total_records = self._count_records(input_path)
        else:
            total_records = len(selected_indices)

        with open(input_path, "r", encoding="utf-8") as fin, open(
            output_path, "w", encoding="utf-8"
        ) as fout, tqdm(total=total_records, desc="Labeling RAUQ", unit="record") as progress:
            for idx, line in enumerate(fin):
                if selected_indices is not None and idx not in selected_indices:
                    continue
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                result = self.process_record(record)
                fout.write(json.dumps(result) + "\n")
                fout.flush()
                progress.update(1)

    def process_record(self, record: Dict) -> Dict:
        prompt = self.prompt_builder.build(record)
        prompt_ids = self.model_adapter.encode(prompt)
        prompt_mask = torch.ones_like(prompt_ids, dtype=torch.long, device=self.model_adapter.device)

        if self.max_new_tokens <= 0:
            pred_text = ""
            pred_eval = self._make_pred_eval(record, pred_text)
            return {
                "id": record.get("id"),
                "dataset": record.get("dataset"),
                "prompt": prompt,
                "pred_text": pred_text,
                "pred_eval": pred_eval,
                "correct": self.scorer.score(record, pred_eval, raw_pred=pred_text),
                "alpha": self.rauq.alpha,
                "selected_heads": {},
                "layers": [],
                "u_per_layer": {},
                "u_final": 0.0,
                "u_token": [],
                "a_prev_selected": [],
                "logp_token": [],
                "gen_token_ids": [],
            }

        gen_token_ids: List[int] = []
        logp_token: List[float] = []
        a_prev_all_heads: List[Dict[str, List[float]]] = []

        next_token_id, logp, _, past = self.model_adapter.step(
            prompt_ids, past_key_values=None, attention_mask=prompt_mask
        )
        gen_token_ids.append(next_token_id)
        logp_token.append(logp)
        a_prev_all_heads.append({})  # placeholder for token-level attentions
        if self.debug_decode:
            self._debug_log_token(len(gen_token_ids), next_token_id, logp)

        current_input_ids = torch.tensor([[next_token_id]], dtype=torch.long, device=self.model_adapter.device)
        token_mask = self._make_cumulative_mask(past, input_len=current_input_ids.shape[1])
        stop = self._should_stop(next_token_id)

        while len(gen_token_ids) < self.max_new_tokens and not stop:
            next_token_id, logp, attns, past = self.model_adapter.step(
                current_input_ids,
                past_key_values=past,
                attention_mask=token_mask,
            )
            a_prev_all_heads[-1] = self._extract_prev_attention(attns)

            gen_token_ids.append(next_token_id)
            logp_token.append(logp)
            a_prev_all_heads.append({})
            if self.debug_decode:
                self._debug_log_token(len(gen_token_ids), next_token_id, logp)

            current_input_ids = torch.tensor([[next_token_id]], dtype=torch.long, device=self.model_adapter.device)
            token_mask = self._make_cumulative_mask(past, input_len=current_input_ids.shape[1])
            stop = self._should_stop(next_token_id)

        if gen_token_ids and a_prev_all_heads[-1] == {}:
            prev_debug_state = getattr(self.model_adapter, "debug_decode", False)
            if prev_debug_state:
                self.model_adapter.debug_decode = False
            try:
                _, _, attns, _ = self.model_adapter.step(
                    current_input_ids,
                    past_key_values=past,
                    attention_mask=self._make_cumulative_mask(past, input_len=current_input_ids.shape[1]),
                )
            finally:
                if prev_debug_state:
                    self.model_adapter.debug_decode = prev_debug_state
            a_prev_all_heads[-1] = self._extract_prev_attention(attns)

        if gen_token_ids:
            template = a_prev_all_heads[0]
            if not template:
                template = self._zero_attention_template()
            zero_dict = {layer: [0.0] * len(heads) for layer, heads in template.items()}
            a_prev_all_heads[0] = zero_dict

        pred_text = self.model_adapter.tokenizer.decode(gen_token_ids, skip_special_tokens=True)
        pred_eval = self._make_pred_eval(record, pred_text)
        correct = self.scorer.score(record, pred_eval, raw_pred=pred_text)

        selected_heads = self.head_selector.select_for_sequence(a_prev_all_heads)
        layers = sorted(int(layer[1:]) for layer in selected_heads.keys()) if selected_heads else []
        a_prev_selected = self._select_heads(a_prev_all_heads, selected_heads, layers)
        u_per_layer, u_final, u_token = self.rauq.compute(logp_token, a_prev_selected)

        result = {
            "id": record.get("id"),
            "dataset": record.get("dataset"),
            "prompt": prompt,
            "pred_text": pred_text,
            "pred_eval": pred_eval,
            "correct": correct,
            "alpha": self.rauq.alpha,
            "selected_heads": selected_heads,
            "layers": layers,
            "u_per_layer": u_per_layer,
            "u_final": u_final,
            "u_token": u_token,
            "a_prev_selected": a_prev_selected,
            "logp_token": logp_token,
            "gen_token_ids": gen_token_ids,
        }

        if self.store_all_heads:
            result["a_prev_all_heads"] = a_prev_all_heads

        return result

    def _should_stop(self, token_id: int) -> bool:
        eos_ids = self.model_adapter.eos_token_ids
        return bool(eos_ids) and token_id in eos_ids

    def _debug_log_token(self, step_idx: int, token_id: int, logp: float) -> None:
        token_text = self.model_adapter.tokenizer.decode([token_id], skip_special_tokens=False)
        if not token_text:
            pieces = self.model_adapter.tokenizer.convert_ids_to_tokens([token_id])
            token_text = pieces[0] if pieces else str(token_id)
        display = token_text.replace("\n", "\\n")
        print(f"[Runner] step={step_idx} token={display!r} (id={token_id}) logp={logp:.3f}")

    def _extract_prev_attention(self, attentions: tuple) -> Dict[str, List[float]]:
        if attentions is None:
            return self._zero_attention_template()
        attn_per_layer: Dict[str, List[float]] = {}
        for layer_idx, attn in enumerate(attentions):
            layer_attn = attn[0]  # (heads, q_len, k_len)
            heads, q_len, k_len = layer_attn.shape
            if heads == 0:
                attn_per_layer[f"l{layer_idx}"] = []
                continue
            if k_len < 2:
                values = [0.0] * heads
            else:
                prev_idx = k_len - 2
                last_query = q_len - 1
                values = layer_attn[:, last_query, prev_idx].detach().cpu().tolist()
            attn_per_layer[f"l{layer_idx}"] = [float(v) for v in values]
        return attn_per_layer

    def _select_heads(
        self,
        a_prev_all_heads: List[Dict[str, List[float]]],
        selected_heads: Dict[str, int],
        layers: List[int],
    ) -> List[Dict[str, float]]:
        if not a_prev_all_heads:
            return []

        a_prev_selected: List[Dict[str, float]] = []
        for token_heads in a_prev_all_heads:
            layer_values: Dict[str, float] = {}
            for layer_idx in layers:
                key = f"l{layer_idx}"
                head_idx = selected_heads.get(key, 0)
                heads = token_heads.get(key, [])
                value = float(heads[head_idx]) if head_idx < len(heads) else 0.0
                layer_values[key] = value
            a_prev_selected.append(layer_values)
        return a_prev_selected

    def _make_pred_eval(self, record: Dict, pred_text: str) -> str:
        answer_type = (record.get("answer_type") or "open").lower()
        options = record.get("options") or []

        if answer_type == "choice":
            picked = self.scorer.pick_choice(pred_text, options)
            return self.scorer.normalize(picked)

        if answer_type == "boolean":
            norm = self.scorer.normalize(pred_text)
            if norm.startswith("yes"):
                return "yes"
            if norm.startswith("no"):
                return "no"
            return norm.split(" ", 1)[0] if norm else ""

        if answer_type == "numeric":
            numeric = self.scorer.extract_numeric(pred_text)
            if numeric is not None:
                numeric = numeric.strip()
                if numeric.endswith(".0"):
                    numeric = numeric[:-2]
                return numeric
            return self.scorer.normalize(pred_text)

        return self.scorer.normalize(pred_text)

    def _zero_attention_template(self) -> Dict[str, List[float]]:
        num_layers = int(getattr(self.model_adapter.config, "num_hidden_layers", 0) or 0)
        num_heads = int(getattr(self.model_adapter.config, "num_attention_heads", 0) or 0)
        return {f"l{layer}": [0.0] * num_heads for layer in range(num_layers)}

    def _make_cumulative_mask(self, past, input_len: int) -> torch.LongTensor:
        """Return attention mask covering past cache + current input tokens."""
        device = self.model_adapter.device
        if past is None:
            return torch.ones((1, input_len), dtype=torch.long, device=device)

        if hasattr(past, "past_key_values"):
            past = past.past_key_values

        if not past:
            return torch.ones((1, input_len), dtype=torch.long, device=device)

        first_entry = past[0]
        tensor_candidate = None
        if isinstance(first_entry, (tuple, list)):
            for item in first_entry:
                if torch.is_tensor(item) and item.dim() >= 3:
                    tensor_candidate = item
                    break
        elif torch.is_tensor(first_entry):
            tensor_candidate = first_entry

        if tensor_candidate is None:
            raise RuntimeError("Unable to infer past length from past_key_values")

        batch = tensor_candidate.shape[0]
        past_len = tensor_candidate.shape[-2]
        total_len = past_len + input_len
        mask_device = tensor_candidate.device if tensor_candidate.device.type != "meta" else device
        return torch.ones((batch, total_len), dtype=torch.long, device=mask_device)

    def _select_record_indices(self, input_path: str) -> Set[int]:
        indices: List[int] = []
        with open(input_path, "r", encoding="utf-8") as fin:
            for idx, line in enumerate(fin):
                if line.strip():
                    indices.append(idx)

        if not indices:
            return set()

        target_count = int(math.ceil(len(indices) * self.dataset_fraction))
        target_count = max(min(target_count, len(indices)), 0)
        if target_count == len(indices):
            return set(indices)
        if target_count == 0:
            return set()

        chosen = set(random.sample(indices, target_count))
        return chosen

    def _count_records(self, input_path: str) -> int:
        count = 0
        with open(input_path, "r", encoding="utf-8") as fin:
            for line in fin:
                if line.strip():
                    count += 1
        return count


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
