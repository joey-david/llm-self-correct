# scoring.py — minimal, fixed AlignScore import + sane defaults + non-spammy usage
from __future__ import annotations

import logging
import os
import re
from typing import Dict, Iterable, List, Optional
from dotenv import load_dotenv

load_dotenv(override=False)

# --- FIXED IMPORT: try the public symbol first, then the submodule path ---
_AlignScoreImpl = None
try:
    from alignscore import AlignScore as _AlignScoreImpl  # type: ignore
except Exception:
    try:
        from alignscore.alignscore import AlignScore as _AlignScoreImpl  # type: ignore
    except Exception:
        _AlignScoreImpl = None

_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")


class AnswerScorer:
    """
    Heuristic + (optional) AlignScore-based answer scorer.
    - If AlignScore is installed and a model is provided (or defaulted), we use it.
    - Otherwise we fall back to lexical heuristics only (silently, no spam).
    """

    def __init__(
        self,
        alignscore_model: Optional[str] = None,
        alignscore_threshold: float = 0.5,
        alignscore_device: Optional[str] = None,
        alignscore_ckpt: Optional[str] = None,
        alignscore_batch_size: Optional[int] = None,
        alignscore_eval_mode: Optional[str] = None,
        # If True, we will try very hard to turn AlignScore on (with a sane default model).
        # If False, AlignScore is disabled unless a model is explicitly provided.
        alignscore_auto_enable: bool = True,
    ) -> None:
        # --- Resolve config/env with sane defaults ---
        env = os.environ
        self.alignscore_model = alignscore_model or env.get("ALIGNSCORE_MODEL")
        self.alignscore_device = alignscore_device or env.get("ALIGNSCORE_DEVICE") or env.get("PYTORCH_DEVICE") or "cpu"
        self.alignscore_ckpt = alignscore_ckpt or env.get("ALIGNSCORE_CKPT")

        # Default batch size
        if alignscore_batch_size is not None:
            self.alignscore_batch_size = max(int(alignscore_batch_size), 1)
        else:
            try:
                self.alignscore_batch_size = max(int(env.get("ALIGNSCORE_BATCH_SIZE", "8")), 1)
            except ValueError:
                logging.warning("Invalid ALIGNSCORE_BATCH_SIZE; defaulting to 8")
                self.alignscore_batch_size = 8

        # Eval mode (AlignScore supports modes like 'nli_sp', 'nli', etc.)
        self.alignscore_eval_mode = alignscore_eval_mode or env.get("ALIGNSCORE_MODE") or "nli_sp"

        # Threshold
        try:
            self.alignscore_threshold = float(env.get("ALIGNSCORE_THRESHOLD", alignscore_threshold))
        except ValueError:
            logging.warning("Invalid ALIGNSCORE_THRESHOLD; falling back to %.3f", float(alignscore_threshold))
            self.alignscore_threshold = float(alignscore_threshold)

        # If auto-enable is on and user did not set a model, provide a robust default
        # that AlignScore commonly supports (can be overridden via env/arg).
        # You can switch this to 'roberta-large-mnli' if you prefer.
        if alignscore_auto_enable and not self.alignscore_model:
            self.alignscore_model = "facebook/bart-large-mnli"

        # Internal state
        self._alignscore = None
        self._warned_init_failure = False
        self._disabled_reason: Optional[str] = None

        # If the library is missing or we intentionally disabled, record the reason once.
        if _AlignScoreImpl is None:
            self._disabled_reason = (
                "alignscore package not available. Install with: pip install alignscore"
            )
        elif not self.alignscore_model:
            self._disabled_reason = (
                "AlignScore model not set (set ALIGNSCORE_MODEL or pass alignscore_model=...)"
            )

        # One-time info (no per-example spam)
        if self._disabled_reason:
            logging.info("[AnswerScorer] AlignScore disabled: %s", self._disabled_reason)
        else:
            logging.info(
                "[AnswerScorer] AlignScore configured: model=%s, device=%s, batch=%d, mode=%s, ckpt=%s",
                self.alignscore_model, self.alignscore_device, self.alignscore_batch_size,
                self.alignscore_eval_mode, self.alignscore_ckpt or "<none>",
            )

    # -------------------- Public helpers --------------------

    def pick_choice(self, pred_text: str, options: Iterable[str]) -> str:
        options_list = list(options or [])
        if not options_list:
            return (pred_text or "").strip()

        pred_clean = (pred_text or "").strip()
        if not pred_clean:
            return options_list[0]

        # allow "A/B/C/..." selection by first token
        first_token = pred_clean.split(None, 1)[0].lower()
        for idx, opt in enumerate(options_list):
            if first_token == chr(ord("a") + idx):
                return opt

        # exact (case-insensitive)
        for opt in options_list:
            if pred_clean.lower() == (opt or "").strip().lower():
                return opt

        # fallback: token-overlap
        pred_tokens = {t.lower() for t in pred_clean.split()}
        best_idx, best_overlap = 0, -1
        for idx, opt in enumerate(options_list):
            toks = {t.lower() for t in (opt or "").split() if t}
            overlap = len(pred_tokens & toks)
            if overlap > best_overlap:
                best_idx, best_overlap = idx, overlap
        return options_list[best_idx]

    def extract_numeric(self, pred_text: str) -> Optional[str]:
        if pred_text is None:
            return None
        m = _NUM_RE.findall(pred_text)
        return m[-1] if m else None

    def score(self, record: Dict, pred_eval: str, raw_pred: Optional[str] = None) -> bool:
        """Return True if predicted answer is accepted as correct."""
        answers = [a.strip() for a in (record.get("answers") or []) if isinstance(a, str) and a.strip()]
        candidate_text = "" if (raw_pred if raw_pred is not None else pred_eval) is None else str(raw_pred if raw_pred is not None else pred_eval)

        # 1) Try AlignScore (if enabled & available)
        align = self._alignscore_match(record, candidate_text)
        if align is True:
            return True
        if align is False:
            return False
        # align is None => fall back to heuristics

        # 2) Heuristics (exact/subset/containment, case-insensitive)
        if not answers:
            return False
        evaluated = (pred_eval or "").strip()
        if not evaluated:
            return False
        return self._exact_or_subset_match(evaluated, answers)

    # -------------------- Internals --------------------

    def _exact_or_subset_match(self, candidate: str, answers: Iterable[str]) -> bool:
        c = candidate.lower().strip()
        ctoks = {t for t in c.split() if t}
        for a in answers:
            al = a.lower().strip()
            if c == al:
                return True
            if c and c in al:
                return True
            if al and al in c:
                return True
            atoks = {t for t in al.split() if t}
            if ctoks and atoks and (ctoks <= atoks or atoks <= ctoks):
                return True
        return False

    def _alignscore_match(self, record: Dict, pred_text: str) -> Optional[bool]:
        if self._disabled_reason:
            print(self._disabled_reason)
            return None
        pred_text = (pred_text or "").strip()
        if not pred_text:
            return None

        refs = [a.strip() for a in (record.get("answers") or []) if isinstance(a, str) and a.strip()]
        if not refs:
            return None

        # Lazy init once
        try:
            scorer = self._ensure_alignscore()
        except RuntimeError as exc:
            if not self._warned_init_failure:
                logging.warning("[AnswerScorer] AlignScore init failed: %s", exc)
                self._warned_init_failure = True
            return None

        claims = [pred_text] * len(refs)
        try:
            raw = scorer.score(contexts=refs, claims=claims)
        except TypeError:
            raw = scorer.score(refs, claims)

        scores = self._extract_align_scores(raw)
        if not scores:
            return None
        return max(scores) >= self.alignscore_threshold

    def _extract_align_scores(self, payload: object) -> List[float]:
        if payload is None:
            return []
        if isinstance(payload, dict):
            for k in ("scores", "score", "align_scores", "alignscore"):
                v = payload.get(k)  # type: ignore[call-arg]
                if v is not None:
                    return self._as_float_list(v)
            return []
        return self._as_float_list(payload)

    def _as_float_list(self, v: object) -> List[float]:
        if v is None:
            return []
        if isinstance(v, (int, float)):
            return [float(v)]
        if isinstance(v, (list, tuple)):
            out: List[float] = []
            for x in v:
                try:
                    out.append(float(x))
                except (TypeError, ValueError):
                    pass
            return out
        return []

    def _ensure_alignscore(self):
        if self._alignscore is not None:
            return self._alignscore
        if _AlignScoreImpl is None:
            raise RuntimeError("alignscore not installed. pip install alignscore")

        # At this point we *must* have a model name (we set a default earlier if auto-enabled)
        if not self.alignscore_model:
            raise RuntimeError("ALIGNSCORE_MODEL not provided; set env or pass alignscore_model='...'")

        try:
            self._alignscore = _AlignScoreImpl(
                model=self.alignscore_model,
                batch_size=self.alignscore_batch_size,
                device=self.alignscore_device or "cpu",
                ckpt_path=self.alignscore_ckpt,
                evaluation_mode=self.alignscore_eval_mode,
            )
        except Exception as exc:
            raise RuntimeError(f"Failed to initialize AlignScore ({self.alignscore_model}): {exc}") from exc
        return self._alignscore
