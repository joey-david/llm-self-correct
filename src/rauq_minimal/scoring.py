from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional
import spacy


_AlignScoreImpl = None
try:
    from alignscore import AlignScore as _AlignScoreImpl  # type: ignore
except Exception:
    try:
        from alignscore.alignscore import AlignScore as _AlignScoreImpl  # type: ignore
    except Exception:
        _AlignScoreImpl = None

_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")

# Embedded defaults for AlignScore usage (no .env or shell needed)
_DEFAULT_MODEL = "roberta-base"  # default to smaller model by request
_DEFAULT_EVAL_MODE = "nli_sp"
_DEFAULT_BATCH_SIZE = 16
_DEFAULT_THRESHOLD = 0.5

_HF_REPO = "yzha/AlignScore"
_CKPT_BY_BACKBONE = {
    "roberta-large": "AlignScore-large.ckpt",
    "roberta-base": "AlignScore-base.ckpt",
}


class AnswerScorer:
    """
    Heuristic + (optional) AlignScore-based answer scorer.
    - If AlignScore is installed and a model is provided (or defaulted), we use it.
    - Otherwise we fall back to lexical heuristics only (silently, no spam).
    """

    def __init__(
        self,
        alignscore_model: Optional[str] = None,
        alignscore_threshold: Optional[float] = None,
        alignscore_device: Optional[str] = None,
        alignscore_ckpt: Optional[str] = None,
        alignscore_batch_size: Optional[int] = None,
        alignscore_eval_mode: Optional[str] = None,
    ) -> None:
        # Repo root and checkpoints dir
        self._here = Path(__file__).resolve()
        # scoring.py -> rauq_minimal -> src -> repo root
        self._repo_root = self._here.parents[2]
        self._ckpt_dir = self._repo_root / "checkpoints"
        self._ckpt_dir.mkdir(parents=True, exist_ok=True)

        # Embedded defaults
        self.alignscore_model = (alignscore_model or _DEFAULT_MODEL).strip()
        self.alignscore_eval_mode = (alignscore_eval_mode or _DEFAULT_EVAL_MODE).strip()
        self.alignscore_batch_size = int(alignscore_batch_size or _DEFAULT_BATCH_SIZE)
        self.alignscore_threshold = float(
            _DEFAULT_THRESHOLD if alignscore_threshold is None else alignscore_threshold
        )

        # Device detection if not provided
        if alignscore_device:
            self.alignscore_device = alignscore_device
        else:
            try:
                import torch

                self.alignscore_device = "cuda" if torch.cuda.is_available() else "cpu"
            except Exception:
                self.alignscore_device = "cpu"

        # Default checkpoint path based on backbone
        ckpt_name = _CKPT_BY_BACKBONE.get(self.alignscore_model, _CKPT_BY_BACKBONE[_DEFAULT_MODEL])
        default_ckpt_path = self._ckpt_dir / ckpt_name
        self.alignscore_ckpt = str(alignscore_ckpt) if alignscore_ckpt else str(default_ckpt_path)

        self._alignscore = None
        self._warned_init_failure = False
        self._disabled_reason: Optional[str] = None if _AlignScoreImpl is not None else (
            "alignscore package not available. Install with: pip install alignscore-SpeedOfMagic"
        )

        # One-time info
        if self._disabled_reason:
            logging.info("[AnswerScorer] AlignScore disabled: %s", self._disabled_reason)
        else:
            logging.info(
                "[AnswerScorer] AlignScore configured: model=%s, device=%s, batch=%d, mode=%s, ckpt=%s",
                self.alignscore_model,
                self.alignscore_device,
                self.alignscore_batch_size,
                self.alignscore_eval_mode,
                self.alignscore_ckpt,
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
            print(f"[AnswerScorer] AlignScore disabled: {self._disabled_reason}")
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
            print(f"[AnswerScorer] AlignScore init failed: {exc}")
            # Avoid re-attempting every record; mark disabled for this run
            self._disabled_reason = f"init failed: {exc}"
            return None

        claims = [pred_text] * len(refs)
        rec_id = record.get("id", "<unknown>")
        print(f"[AnswerScorer] Using AlignScore for record {rec_id} with {len(refs)} references")
        try:
            raw = scorer.score(contexts=refs, claims=claims)
        except TypeError:
            raw = scorer.score(refs, claims)

        scores = self._extract_align_scores(raw)
        if not scores:
            return None
        best = max(scores)
        print(
            f"[AnswerScorer] AlignScore best score={best:.4f} (threshold={self.alignscore_threshold:.4f}) for {rec_id}"
        )
        return best >= self.alignscore_threshold

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
            raise RuntimeError("AlignScore backbone not provided; pass alignscore_model='roberta-large' or 'roberta-base'")

        # Ensure spaCy English model is available (AlignScore requires it)
        try:
            self._ensure_spacy_en()
        except Exception as spacy_exc:
            raise RuntimeError(
                f"spaCy 'en_core_web_sm' model missing and auto-download failed: {spacy_exc}"
            )

        # Ensure checkpoint exists: attempt download if missing
        ckpt_path = Path(self.alignscore_ckpt)
        if not ckpt_path.is_file():
            print(f"[AnswerScorer] Checkpoint missing at {ckpt_path}. Attempting download...")
            self._download_ckpt(ckpt_path, self.alignscore_model)
            if not ckpt_path.is_file():
                raise RuntimeError(f"Checkpoint not found after download attempt: {ckpt_path}")

        try:
            # Temporarily silence third-party loggers (Transformers/PL) to hide
            # benign init warnings like uninitialized pooler weights and PL checkpoint upgrade.
            _hf_logging = None
            _prev_hf_level = None
            _pl_logger = None
            _prev_pl_level = None
            try:
                from transformers.utils import logging as _hf_logging  # type: ignore

                _prev_hf_level = _hf_logging.get_verbosity()
                _hf_logging.set_verbosity_error()
            except Exception:
                _hf_logging = None  # type: ignore
                _prev_hf_level = None
            try:
                import logging as _pylogging

                _pl_logger = _pylogging.getLogger("pytorch_lightning")
                _prev_pl_level = _pl_logger.level
                _pl_logger.setLevel(_pylogging.ERROR)
            except Exception:
                _pl_logger = None
                _prev_pl_level = None

            try:
                self._alignscore = _AlignScoreImpl(
                    model=self.alignscore_model,
                    batch_size=self.alignscore_batch_size,
                    device=self.alignscore_device or "cpu",
                    ckpt_path=self.alignscore_ckpt,
                    evaluation_mode=self.alignscore_eval_mode,
                )
            finally:
                try:
                    if _hf_logging is not None and _prev_hf_level is not None:
                        _hf_logging.set_verbosity(_prev_hf_level)
                except Exception:
                    pass
                try:
                    if _pl_logger is not None and _prev_pl_level is not None:
                        _pl_logger.setLevel(_prev_pl_level)
                except Exception:
                    pass
            print(
                "[AnswerScorer] Initialized AlignScore",
                f"model={self.alignscore_model}",
                f"ckpt={self.alignscore_ckpt}",
                f"device={self.alignscore_device}",
                f"mode={self.alignscore_eval_mode}",
                f"batch={self.alignscore_batch_size}",
            )
        except Exception as exc:
            raise RuntimeError(f"Failed to initialize AlignScore ({self.alignscore_model}): {exc}") from exc
        return self._alignscore

    def _ensure_spacy_en(self) -> None:
        """Ensure spaCy and the 'en_core_web_sm' model are available."""
        try:
            import spacy  # type: ignore
        except Exception as exc:
            raise RuntimeError("spaCy is not installed") from exc

        try:
            spacy.load("en_core_web_sm")
            return
        except Exception:
            print("[AnswerScorer] spaCy model 'en_core_web_sm' not found; attempting download...")
            try:
                from spacy.cli import download as spacy_download  # type: ignore

                spacy_download("en_core_web_sm")
                # Verify
                spacy.load("en_core_web_sm")
                print("[AnswerScorer] spaCy model 'en_core_web_sm' installed.")
                return
            except Exception as exc:
                raise RuntimeError(f"Failed to download/load 'en_core_web_sm': {exc}") from exc

    def _download_ckpt(self, dest: Path, backbone: str) -> None:
        """Download AlignScore checkpoint to `dest` (HF hub preferred, HTTPS fallback)."""
        dest.parent.mkdir(parents=True, exist_ok=True)
        ckpt_name = _CKPT_BY_BACKBONE.get(backbone, _CKPT_BY_BACKBONE[_DEFAULT_MODEL])

        # Try huggingface_hub first
        try:
            from huggingface_hub import hf_hub_download  # type: ignore
            import shutil

            print(
                f"[AnswerScorer] Downloading via huggingface_hub: repo={_HF_REPO}, file={ckpt_name}"
            )
            # Place the file directly under our checkpoints dir (no symlinks)
            downloaded = hf_hub_download(
                repo_id=_HF_REPO,
                filename=ckpt_name,
                local_dir=str(dest.parent),
            )
            downloaded_path = Path(downloaded)
            # Ensure final path matches expected dest (robust across FS boundaries)
            if downloaded_path.resolve() != dest.resolve():
                try:
                    shutil.move(str(downloaded_path), str(dest))
                except Exception:
                    shutil.copy2(str(downloaded_path), str(dest))
            # Double-check file exists
            if not dest.is_file():
                raise RuntimeError(f"Failed to place checkpoint at {dest}")
            return
        except Exception as hub_exc:
            print(f"[AnswerScorer] huggingface_hub unavailable or failed: {hub_exc}")

        # Fallback to direct HTTPS
        url = f"https://huggingface.co/{_HF_REPO}/resolve/main/{ckpt_name}"
        print(f"[AnswerScorer] Downloading via HTTPS: {url}")
        try:
            import urllib.request

            with urllib.request.urlopen(url) as resp, open(dest, "wb") as fout:
                fout.write(resp.read())
        except Exception as http_exc:
            print(f"[AnswerScorer] HTTP download failed: {http_exc}")
