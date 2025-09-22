"""Learn a RAUQ trigger threshold via logistic calibration."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, List, Sequence, Tuple

import numpy as np
import torch

try:  # pragma: no cover - optional dependency
    from sklearn.linear_model import LogisticRegression
except ImportError:  # pragma: no cover
    LogisticRegression = None

from .config import RAUQConfig, ThresholdTrainingConfig
from .data.utils import load_prompt_completion_pairs
from .rauq import RAUQScorer
from .utils.logging import setup_logging
from .utils.model import LoadedModel, load_model

logger = setup_logging(name=__name__)

SequenceMetric = Callable[[str, str, str], bool]


def exact_match_metric(prompt: str, prediction: str, reference: str) -> bool:
    return prediction.strip() == reference.strip()


@dataclass
class ThresholdResult:
    theta: float
    logistic_coef: float
    logistic_intercept: float

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "theta": self.theta,
            "logistic_coef": self.logistic_coef,
            "logistic_intercept": self.logistic_intercept,
        }
        path.write_text(json.dumps(payload, indent=2))
        logger.info("Saved threshold to %s", path)

    @classmethod
    def load(cls, path: Path) -> "ThresholdResult":
        payload = json.loads(path.read_text())
        return cls(theta=payload["theta"], logistic_coef=payload["logistic_coef"], logistic_intercept=payload["logistic_intercept"])


class ThresholdTrainer:
    def __init__(
        self,
        loaded: LoadedModel,
        scorer: RAUQScorer,
        metric_fn: SequenceMetric,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ) -> None:
        self.model = loaded.model
        self.tokenizer = loaded.tokenizer
        self.metric_fn = metric_fn
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.device = self.model.device
        self.scorer = scorer

    def _sample(self, logits: torch.Tensor) -> torch.Tensor:
        if self.temperature == 0.0:
            return torch.argmax(logits, dim=-1)
        if self.top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            probs = torch.softmax(sorted_logits / self.temperature, dim=-1)
            cumulative = probs.cumsum(dim=-1)
            mask = cumulative > self.top_p
            mask[..., 1:] = mask[..., :-1].clone()
            mask[..., 0] = False
            sorted_logits = sorted_logits.masked_fill(mask, float("-inf"))
            probs = torch.softmax(sorted_logits / self.temperature, dim=-1)
            choice = torch.multinomial(probs, num_samples=1)
            return sorted_indices.gather(-1, choice).squeeze(-1)
        probs = torch.softmax(logits / self.temperature, dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)

    def _decode(self, prompt: str) -> Tuple[str, List[float]]:
        tokenizer = self.tokenizer
        model = self.model
        scorer = self.scorer
        scorer.reset()

        encoded = tokenizer(prompt, return_tensors="pt")
        input_ids = encoded["input_ids"].to(model.device)
        attention_mask = encoded["attention_mask"].to(model.device)
        prompt_length = input_ids.size(1)

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=True,
                output_attentions=False,
            )
        past_key_values = outputs.past_key_values
        generated = input_ids
        scores: List[float] = []

        for _ in range(self.max_new_tokens):
            with torch.no_grad():
                step_outputs = model(
                    input_ids=generated[:, -1:],
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    use_cache=True,
                    output_attentions=False,
                )
            logits = step_outputs.logits[:, -1, :]
            next_token = self._sample(logits)

            generated = torch.cat([generated, next_token.unsqueeze(-1)], dim=-1)
            attention_mask = torch.ones_like(generated, device=model.device)

            with torch.no_grad():
                attn_outputs = model(
                    input_ids=next_token.unsqueeze(-1),
                    attention_mask=attention_mask,
                    past_key_values=step_outputs.past_key_values,
                    use_cache=True,
                    output_attentions=True,
                )
            past_key_values = attn_outputs.past_key_values
            attentions = attn_outputs.attentions

            score = scorer.score_token(
                logits=logits.squeeze(0),
                token_id=next_token.squeeze(0),
                attentions=attentions,
            )
            scores.append(score)

            if next_token.item() == tokenizer.eos_token_id:
                break

        completion_ids = generated[:, prompt_length:]
        completion_text = tokenizer.decode(completion_ids[0], skip_special_tokens=True)
        return completion_text, scores

    def collect(self, samples: Sequence[Tuple[str, str]]) -> Tuple[np.ndarray, np.ndarray]:
        token_scores: List[float] = []
        labels: List[int] = []
        for prompt, reference in samples:
            prediction, scores = self._decode(prompt)
            is_correct = self.metric_fn(prompt, prediction, reference)
            failure = 0 if is_correct else 1
            token_scores.extend(scores)
            labels.extend([failure] * len(scores))
        return np.array(token_scores), np.array(labels)

    def fit_threshold(self, scores: np.ndarray, labels: np.ndarray, penalty: float = 1.0) -> ThresholdResult:
        if LogisticRegression is None:
            raise RuntimeError("scikit-learn is required for threshold calibration")
        clf = LogisticRegression(C=penalty, max_iter=1000)
        clf.fit(scores.reshape(-1, 1), labels)
        coef = float(clf.coef_[0][0])
        intercept = float(clf.intercept_[0])

        thresholds = np.unique(scores)
        best_theta = thresholds[0]
        best_score = -np.inf
        for theta in thresholds:
            preds = scores >= theta
            tp = np.logical_and(preds == 1, labels == 1).sum()
            fn = np.logical_and(preds == 0, labels == 1).sum()
            fp = np.logical_and(preds == 1, labels == 0).sum()
            tn = np.logical_and(preds == 0, labels == 0).sum()
            tpr = tp / (tp + fn + 1e-12)
            fpr = fp / (fp + tn + 1e-12)
            youden = tpr - fpr
            if youden > best_score:
                best_score = youden
                best_theta = theta
        return ThresholdResult(theta=float(best_theta), logistic_coef=coef, logistic_intercept=intercept)


def train_threshold(config: ThresholdTrainingConfig, metric_fn: SequenceMetric = exact_match_metric) -> ThresholdResult:
    samples = load_prompt_completion_pairs(config.calibration_paths)
    if config.max_samples is not None:
        samples = samples[: config.max_samples]
    if not samples:
        raise ValueError("No calibration samples provided")

    loaded = load_model(
        model_name=config.model_name,
        tokenizer_name=config.tokenizer_name,
        device=config.device,
    )

    scorer = RAUQScorer.from_config(
        RAUQConfig(
            alpha=config.alpha,
            head_indices_path=config.head_indices_path,
            device=config.device,
        ),
        num_layers=loaded.model.config.num_hidden_layers,
    )
    trainer = ThresholdTrainer(
        loaded=loaded,
        scorer=scorer,
        metric_fn=metric_fn,
    )

    scores, labels = trainer.collect(samples)
    result = trainer.fit_threshold(scores, labels, penalty=config.logistic_penalty)
    result.save(config.output_path)
    return result


__all__ = ["train_threshold", "ThresholdTrainer", "ThresholdResult", "exact_match_metric"]
