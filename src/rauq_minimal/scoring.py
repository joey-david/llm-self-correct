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


_PUNCT_RE = re.compile(r"[\W_]+", re.UNICODE)
_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")


class AnswerScorer:
    """Utility helpers for normalizing predictions and scoring."""

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

    def normalize(self, text: str) -> str:
        if text is None:
            return ""
        text = text.strip().lower()
        text = _PUNCT_RE.sub(" ", text)
        return re.sub(r"\s+", " ", text).strip()

    def pick_choice(self, pred_text: str, options: Iterable[str]) -> str:
        options_list = list(options or [])
        if not options_list:
            return pred_text.strip()

        pred_norm = self.normalize(pred_text)
        # Try to detect leading letter selection (A/B/...)
        first_token = pred_norm.split(" ", 1)[0] if pred_norm else ""
        if first_token:
            for idx, opt in enumerate(options_list):
                label = chr(ord("a") + idx)
                if first_token == label:
                    return opt

        option_norms = [self.normalize(opt) for opt in options_list]
        if pred_norm in option_norms:
            return options_list[option_norms.index(pred_norm)]

        pred_tokens = set(pred_norm.split())
        best_idx = 0
        best_overlap = -1
        for idx, opt_norm in enumerate(option_norms):
            opt_tokens = set(opt_norm.split())
            if not opt_tokens:
                continue
            overlap = len(pred_tokens & opt_tokens)
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
        answer_type = (record.get("answer_type") or "open").lower()
        answers_norm = record.get("answers_normalized") or []
        pred_norm = self.normalize(pred_eval)

        if answer_type != "open":
            if not answers_norm:
                return False
            return self._heuristic_match(answers_norm, pred_norm)

        heuristic_match = self._heuristic_match(answers_norm, pred_norm) if answers_norm else False
        if heuristic_match:
            return True

        candidate = raw_pred if raw_pred is not None else pred_eval
        return self._alignscore_match(record, candidate)

    def _heuristic_match(self, answers_norm: Iterable[str], pred_norm: str) -> bool:
        if not answers_norm or not pred_norm:
            return False
        return pred_norm in {a.lower() for a in answers_norm}

    def _alignscore_match(self, record: Dict, pred_text: str) -> bool:
        if not self.alignscore_model:
            return False
        if not pred_text:
            return False
        answers = [a for a in (record.get("answers") or []) if isinstance(a, str) and a.strip()]
        if not answers:
            return False

        try:
            scorer = self._ensure_alignscore()
        except RuntimeError as exc:
            if not self._alignscore_warned:
                logging.warning("%s", exc)
                self._alignscore_warned = True
            return False

        if not hasattr(scorer, "score"):
            if not self._alignscore_warned:
                logging.warning("AlignScore implementation does not expose a 'score' method; skipping AlignScore check")
                self._alignscore_warned = True
            return False

        contexts = answers
        claims = [pred_text] * len(answers)

        try:
            raw_scores = scorer.score(contexts=contexts, claims=claims)
        except TypeError:
            raw_scores = scorer.score(contexts, claims)

        scores = self._extract_align_scores(raw_scores)
        if not scores:
            return False
        best = max(scores)
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
        except Exception as exc:  # pragma: no cover - depends on external package
            raise RuntimeError(f"Failed to initialize AlignScore: {exc}") from exc
        return self._alignscore
