from __future__ import annotations

import logging
import os
import re
from typing import Dict, Iterable, List, Optional

try:  # load .env so AlignScore picks up Hugging Face tokens automatically
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None
else:
    load_dotenv(override=False)

try:  # optional AlignScore dependency
    from alignscore.alignscore import AlignScore as _AlignScoreImpl  # type: ignore
except ImportError:  # pragma: no cover - alignscore isn't bundled by default
    _AlignScoreImpl = None


_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")


class AnswerScorer:
    """Utility helpers for scoring predictions."""

    def __init__(
        self,
        alignscore_model: Optional[str] = None,
        alignscore_threshold: float = 0.5,
        alignscore_device: Optional[str] = None,
        alignscore_ckpt: Optional[str] = None,
        alignscore_batch_size: Optional[int] = None,
        alignscore_eval_mode: Optional[str] = None,
    ) -> None:
        self.alignscore_model = alignscore_model or os.environ.get("ALIGNSCORE_MODEL")
        self.alignscore_device = alignscore_device or os.environ.get("ALIGNSCORE_DEVICE")
        self.alignscore_ckpt = alignscore_ckpt or os.environ.get("ALIGNSCORE_CKPT")

        batch_override = os.environ.get("ALIGNSCORE_BATCH_SIZE")
        if alignscore_batch_size is not None:
            self.alignscore_batch_size = max(int(alignscore_batch_size), 1)
        elif batch_override is not None:
            try:
                self.alignscore_batch_size = max(int(batch_override), 1)
            except ValueError:
                logging.warning("Invalid ALIGNSCORE_BATCH_SIZE '%s'; defaulting to 8", batch_override)
                self.alignscore_batch_size = 8
        else:
            self.alignscore_batch_size = 8

        self.alignscore_eval_mode = (
            alignscore_eval_mode
            or os.environ.get("ALIGNSCORE_MODE")
            or "nli_sp"
        )

        if not self.alignscore_device:
            self.alignscore_device = os.environ.get("PYTORCH_DEVICE", "cpu")

        threshold_override = os.environ.get("ALIGNSCORE_THRESHOLD")
        if threshold_override is not None:
            try:
                self.alignscore_threshold = float(threshold_override)
            except ValueError:
                logging.warning(
                    "Invalid ALIGNSCORE_THRESHOLD '%s'; falling back to %.3f",
                    threshold_override,
                    float(alignscore_threshold),
                )
                self.alignscore_threshold = float(alignscore_threshold)
        else:
            self.alignscore_threshold = float(alignscore_threshold)

        self._alignscore = None
        self._alignscore_warned = False

    def pick_choice(self, pred_text: str, options: Iterable[str]) -> str:
        options_list = list(options or [])
        if not options_list:
            return (pred_text or "").strip()

        pred_clean = (pred_text or "").strip()
        if not pred_clean:
            return options_list[0]

        first_token = pred_clean.split(None, 1)[0].lower()
        for idx, opt in enumerate(options_list):
            label = chr(ord("a") + idx)
            if first_token == label:
                return opt

        for opt in options_list:
            if pred_clean.lower() == (opt or "").strip().lower():
                return opt

        pred_tokens = {token.lower() for token in pred_clean.split()}
        best_idx = 0
        best_overlap = -1
        for idx, option in enumerate(options_list):
            option_str = (option or "").strip()
            option_tokens = {tok.lower() for tok in option_str.split() if tok}
            if not option_tokens:
                continue
            overlap = len(pred_tokens & option_tokens)
            if overlap > best_overlap:
                best_overlap = overlap
                best_idx = idx
        return options_list[best_idx]

    def extract_numeric(self, pred_text: str) -> Optional[str]:
        if pred_text is None:
            return None
        matches = _NUM_RE.findall(pred_text)
        if not matches:
            return None
        return matches[-1]

    def score(self, record: Dict, pred_eval: str, raw_pred: Optional[str] = None) -> bool:
        answers = [
            a.strip()
            for a in (record.get("answers") or [])
            if isinstance(a, str) and a.strip()
        ]
        candidate_obj = raw_pred if raw_pred is not None else pred_eval
        candidate_text = "" if candidate_obj is None else str(candidate_obj)

        alignscore_result = self._alignscore_match(record, candidate_text)
        if alignscore_result is not None:
            if alignscore_result:
                return True

        if not answers:
            return False

        evaluated = (pred_eval or "").strip()
        if not evaluated:
            return False

        if alignscore_result is None and self._exact_or_subset_match(evaluated, answers):
            return True

        if alignscore_result is False:
            return False

        return False

    def _exact_or_subset_match(self, candidate: str, answers: Iterable[str]) -> bool:
        candidate_lower = candidate.lower()
        candidate_tokens = {tok for tok in candidate_lower.split() if tok}

        for answer in answers:
            answer_lower = answer.lower()
            if candidate_lower == answer_lower:
                return True

            if candidate_lower and candidate_lower in answer_lower:
                return True
            if answer_lower and answer_lower in candidate_lower:
                return True

            answer_tokens = {tok for tok in answer_lower.split() if tok}
            if not candidate_tokens or not answer_tokens:
                continue
            if candidate_tokens <= answer_tokens:
                return True
            if answer_tokens <= candidate_tokens:
                return True

        return False

    def _alignscore_match(self, record: Dict, pred_text: str) -> Optional[bool]:
        if not self.alignscore_model:
            print("[AnswerScorer] AlignScore model not set; skipping AlignScore scoring")
            return None
        if not isinstance(pred_text, str):
            pred_text = str(pred_text)
        pred_text = pred_text.strip()
        if not pred_text:
            return None
        answers = [
            a.strip()
            for a in (record.get("answers") or [])
            if isinstance(a, str) and a.strip()
        ]
        if not answers:
            return None

        try:
            scorer = self._ensure_alignscore()
        except RuntimeError as exc:
            if not self._alignscore_warned:
                logging.warning("%s", exc)
                self._alignscore_warned = True
            print(f"[AnswerScorer] Failed to initialize AlignScore: {exc}")
            return None

        if not hasattr(scorer, "score"):
            if not self._alignscore_warned:
                logging.warning("AlignScore implementation does not expose a 'score' method; skipping AlignScore check")
                self._alignscore_warned = True
            print("[AnswerScorer] AlignScore implementation missing 'score'; skipping")
            return None

        contexts = answers
        claims = [pred_text] * len(answers)

        record_id = record.get("id", "<unknown>")
        print(f"[AnswerScorer] Running AlignScore for record {record_id} with {len(answers)} references")

        try:
            raw_scores = scorer.score(contexts=contexts, claims=claims)
        except TypeError:
            raw_scores = scorer.score(contexts, claims)

        scores = self._extract_align_scores(raw_scores)
        if not scores:
            print(f"[AnswerScorer] AlignScore produced no scores for record {record_id}")
            return None
        best = max(scores)
        print(f"[AnswerScorer] AlignScore best score={best:.4f} (threshold={self.alignscore_threshold:.4f}) for record {record_id}")
        return best >= self.alignscore_threshold

    def _extract_align_scores(self, payload: object) -> List[float]:
        if payload is None:
            return []
        if isinstance(payload, dict):
            for key in ("scores", "score", "align_scores", "alignscore"):
                val = payload.get(key)  # type: ignore[call-arg]
                if val is not None:
                    return self._as_float_list(val)
            return []
        return self._as_float_list(payload)

    def _as_float_list(self, value: object) -> List[float]:
        if value is None:
            return []
        if isinstance(value, (int, float)):
            return [float(value)]
        if isinstance(value, (list, tuple)):
            floats: List[float] = []
            for item in value:
                try:
                    floats.append(float(item))
                except (TypeError, ValueError):
                    continue
            return floats
        return []

    def _ensure_alignscore(self):
        if self._alignscore is not None:
            return self._alignscore
        if _AlignScoreImpl is None:
            raise RuntimeError(
                "alignscore-SpeedOfMagic is required for AlignScore evaluation; install it via pip"
            )

        if not self.alignscore_model:
            raise RuntimeError("AlignScore requires ALIGNSCORE_MODEL to be set")

        if not self.alignscore_ckpt and not self._alignscore_warned:
            logging.warning(
                "ALIGNSCORE_CKPT not provided; initializing AlignScore without pretrained weights"
            )
            self._alignscore_warned = True

        try:
            self._alignscore = _AlignScoreImpl(
                model=self.alignscore_model,
                batch_size=self.alignscore_batch_size,
                device=self.alignscore_device or "cpu",
                ckpt_path=self.alignscore_ckpt,
                evaluation_mode=self.alignscore_eval_mode,
            )
            print(
                "[AnswerScorer] Initialized AlignScore",
                f"model={self.alignscore_model}",
                f"ckpt={self.alignscore_ckpt or '<none>'}",
                f"device={self.alignscore_device}",
                f"mode={self.alignscore_eval_mode}",
            )
        except Exception as exc:  # pragma: no cover - depends on external package
            raise RuntimeError(f"Failed to initialize AlignScore: {exc}") from exc
        return self._alignscore
