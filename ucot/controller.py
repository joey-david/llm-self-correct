"""RAUQ-triggered decoding controller with rollback + micro-CoT."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from .config import ControllerConfig
from .uncertainty import TokenUncertaintyScorer
from .threshold import ThresholdResult
from .utils.logging import setup_logging
from .utils.model import LoadedModel

logger = setup_logging(name=__name__)


@dataclass
class TriggerEvent:
    position: int
    rauq: float
    reason: str = "threshold"


@dataclass
class ControllerOutput:
    completion: str
    completion_tokens: List[int]
    rauq_scores: List[float]
    trigger_events: List[TriggerEvent] = field(default_factory=list)
    total_tokens: int = 0
    extra_tokens: int = 0
    stats: Dict[str, float] = field(default_factory=dict)


class RAUQController:
    """Implements rollback + micro-CoT decoding triggered by RAUQ spikes."""

    def __init__(
        self,
        loaded: LoadedModel,
        config: ControllerConfig,
        scorer: TokenUncertaintyScorer,
        threshold: Optional[ThresholdResult] = None,
    ) -> None:
        self.model = loaded.model
        self.tokenizer = loaded.tokenizer
        self.config = config
        self.device = self.model.device
        self.scorer = scorer
        self.theta = config.theta or (threshold.theta if threshold else None)
        if self.theta is None:
            raise ValueError("Provide `theta` in ControllerConfig or a ThresholdResult.")
        self.rollback_cfg = config.rollback
        self.cot_cfg = config.cot
        self.temperature = config.temperature
        self.top_p = config.top_p
        self.max_new_tokens = config.max_new_tokens
        self.stop_sequences = config.stop_sequences or []
        prefix_ids = self.tokenizer.encode(self.cot_cfg.cot_prefix, add_special_tokens=False)
        self._cot_prefix_ids = torch.tensor(prefix_ids, dtype=torch.long, device=self.device)

    # ------------------------------------------------------------------ helpers

    def _build_state_from_tokens(self, tokens: torch.Tensor) -> Dict[str, torch.Tensor]:
        attention_mask = torch.ones_like(tokens, device=self.device)
        with torch.no_grad():
            outputs = self.model(
                input_ids=tokens,
                attention_mask=attention_mask,
                use_cache=True,
                output_attentions=False,
            )
        return {
            "tokens": tokens,
            "attention_mask": attention_mask,
            "past_key_values": outputs.past_key_values,
            "next_logits": outputs.logits[:, -1, :],
        }

    def _encode_prompt(self, prompt: str) -> Dict[str, torch.Tensor]:
        encoded = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        return self._build_state_from_tokens(encoded["input_ids"])

    def _sample(self, logits: torch.Tensor, temperature: float, top_p: float) -> torch.Tensor:
        if temperature == 0.0:
            return torch.argmax(logits, dim=-1)
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            probs = torch.softmax(sorted_logits / temperature, dim=-1)
            cumulative = probs.cumsum(dim=-1)
            mask = cumulative > top_p
            mask[..., 1:] = mask[..., :-1].clone()
            mask[..., 0] = False
            sorted_logits = sorted_logits.masked_fill(mask, float("-inf"))
            probs = torch.softmax(sorted_logits / temperature, dim=-1)
            choice = torch.multinomial(probs, num_samples=1)
            return sorted_indices.gather(-1, choice).squeeze(-1)
        probs = torch.softmax(logits / temperature, dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)

    def _generate_token(
        self,
        state: Dict[str, torch.Tensor],
        scorer: RAUQScorer,
        temperature: float,
        top_p: float,
        forced_token: Optional[int] = None,
        update_scorer: bool = True,
    ) -> Tuple[int, Optional[float]]:
        logits = state.pop("next_logits", None)
        current_past = state["past_key_values"]
        if logits is None:
            with torch.no_grad():
                outputs = self.model(
                    input_ids=state["tokens"][:, -1:],
                    attention_mask=state["attention_mask"],
                    past_key_values=current_past,
                    use_cache=True,
                    output_attentions=False,
                )
            logits = outputs.logits[:, -1, :]
            current_past = outputs.past_key_values
        if forced_token is None:
            token = self._sample(logits, temperature=temperature, top_p=top_p)
        else:
            token = torch.tensor(
                [forced_token],
                device=state["tokens"].device,
                dtype=state["tokens"].dtype,
            )
        state["tokens"] = torch.cat([state["tokens"], token.unsqueeze(-1)], dim=-1)
        state["attention_mask"] = torch.ones_like(state["tokens"])
        with torch.no_grad():
            attn_outputs = self.model(
                input_ids=token.unsqueeze(-1),
                attention_mask=state["attention_mask"],
                past_key_values=current_past,
                use_cache=True,
                output_attentions=True,
            )
        state["past_key_values"] = attn_outputs.past_key_values
        state["next_logits"] = attn_outputs.logits[:, -1, :]
        if update_scorer:
            score = scorer.score_token(
                logits=logits.squeeze(0),
                token_id=token.squeeze(0),
                attentions=attn_outputs.attentions,
                update_state=True,
            )
        else:
            score = None
        return token.item(), score

    def _should_stop_token(self, token_id: int) -> bool:
        if token_id == self.tokenizer.eos_token_id:
            return True
        decoded = self.tokenizer.decode([token_id], skip_special_tokens=False)
        return any(decoded.endswith(stop) for stop in self.stop_sequences)

    def _remaining_budget(self, state: Dict[str, torch.Tensor], prompt_length: int) -> int:
        return self.max_new_tokens - (state["tokens"].size(1) - prompt_length)

    # -------------------------------------------------------------- cot module

    def _run_cot_candidates(
        self,
        base_tokens: torch.Tensor,
        base_scorer: RAUQScorer,
        prompt_length: int,
    ) -> Tuple[Dict[str, torch.Tensor], RAUQScorer, List[float], List[RAUQScorer]]:
        best_obj = float("inf")
        best_state = None
        best_scorer = None
        best_scores: List[float] = []
        best_snapshots: List[RAUQScorer] = []

        for _ in range(self.cot_cfg.candidates):
            state = self._build_state_from_tokens(base_tokens.clone())
            scorer = base_scorer.clone()
            appended_scores: List[float] = []
            snapshots: List[RAUQScorer] = []

            for prefix_id in self._cot_prefix_ids.tolist():
                self._generate_token(
                    state,
                    scorer,
                    temperature=0.0,
                    top_p=1.0,
                    forced_token=prefix_id,
                    update_scorer=False,
                )
                appended_scores.append(float("nan"))
                snapshots.append(scorer.clone())
                if self._remaining_budget(state, prompt_length) <= 0:
                    break

            cot_lengths = 0
            for _ in range(self.cot_cfg.max_cot_tokens):
                if self._remaining_budget(state, prompt_length) <= 0:
                    break
                token_id, score = self._generate_token(
                    state,
                    scorer,
                    temperature=self.cot_cfg.temperature,
                    top_p=self.cot_cfg.top_p,
                    update_scorer=True,
                )
                if score is not None:
                    appended_scores.append(score)
                    snapshots.append(scorer.clone())
                    cot_lengths += 1
                if token_id == self.tokenizer.eos_token_id:
                    break

            lookahead_scores: List[float] = []
            for _ in range(self.cot_cfg.lookahead_horizon):
                if self._remaining_budget(state, prompt_length) <= 0:
                    break
                token_id, score = self._generate_token(
                    state,
                    scorer,
                    temperature=0.0,
                    top_p=1.0,
                    update_scorer=True,
                )
                if score is not None:
                    appended_scores.append(score)
                    snapshots.append(scorer.clone())
                    lookahead_scores.append(score)
                if token_id == self.tokenizer.eos_token_id:
                    break

            if not lookahead_scores:
                objective = float("inf")
            else:
                avg_uncertainty = sum(lookahead_scores) / len(lookahead_scores)
                objective = avg_uncertainty + self.cot_cfg.length_penalty * cot_lengths

            if objective < best_obj:
                best_obj = objective
                best_state = state
                best_scorer = scorer
                best_scores = appended_scores
                best_snapshots = snapshots

        if best_state is None:
            return self._build_state_from_tokens(base_tokens.clone()), base_scorer, [], []
        return best_state, best_scorer, best_scores, best_snapshots

    # ------------------------------------------------------------------ main API

    def generate(self, prompt: str) -> ControllerOutput:
        state = self._encode_prompt(prompt)
        prompt_length = state["tokens"].size(1)
        rauq_scores: List[float] = []
        trigger_events: List[TriggerEvent] = []
        stable_streak = 0
        anchor = prompt_length
        last_trigger_step = -999
        triggers = 0

        scorer_history: List[TokenUncertaintyScorer] = [self.scorer.clone()]

        while self._remaining_budget(state, prompt_length) > 0:
            token_id, score = self._generate_token(
                state,
                self.scorer,
                temperature=self.temperature,
                top_p=self.top_p,
                update_scorer=True,
            )
            if score is None:
                score = float("nan")
            rauq_scores.append(score)
            scorer_history.append(self.scorer.clone())

            if self._should_stop_token(token_id):
                break

            if score < self.theta:
                stable_streak += 1
                if stable_streak >= self.rollback_cfg.stability_window:
                    anchor = state["tokens"].size(1)
                continue

            stable_streak = 0
            step_index = state["tokens"].size(1) - prompt_length
            if step_index - last_trigger_step < self.rollback_cfg.cooldown:
                continue
            if self.rollback_cfg.max_triggers is not None and triggers >= self.rollback_cfg.max_triggers:
                continue

            triggers += 1
            last_trigger_step = step_index
            trigger_events.append(TriggerEvent(position=step_index, rauq=score))

            rollback_target = max(anchor, state["tokens"].size(1) - self.rollback_cfg.rollback_depth)
            keep_tokens = rollback_target - prompt_length
            base_tokens = state["tokens"][:, :rollback_target]
            state = self._build_state_from_tokens(base_tokens.clone())
            self.scorer = scorer_history[keep_tokens].clone()
            rauq_scores = rauq_scores[:keep_tokens]
            scorer_history = scorer_history[: keep_tokens + 1]

            state, self.scorer, appended_scores, snapshots = self._run_cot_candidates(
                base_tokens=state["tokens"],
                base_scorer=self.scorer,
                prompt_length=prompt_length,
            )
            rauq_scores.extend(appended_scores)
            scorer_history.extend(snapshots)
            anchor = state["tokens"].size(1)

            if not appended_scores:
                break

            if self._should_stop_token(state["tokens"][0, -1].item()):
                break

            if state["tokens"].size(1) - prompt_length >= self.max_new_tokens:
                break

        completion_ids = state["tokens"][0, prompt_length:]
        completion = self.tokenizer.decode(completion_ids, skip_special_tokens=True)
        output = ControllerOutput(
            completion=completion,
            completion_tokens=completion_ids.tolist(),
            rauq_scores=rauq_scores,
            trigger_events=trigger_events,
            total_tokens=state["tokens"].size(1),
            extra_tokens=completion_ids.size(0),
            stats={"triggers": len(trigger_events)},
        )
        return output


__all__ = ["RAUQController", "ControllerOutput", "TriggerEvent"]
