"""Load and persist calibration corpora for RAUQ preparation."""
from __future__ import annotations

from pathlib import Path
from typing import List, Sequence, Tuple

import json
import random
from urllib import request


Sample = Tuple[str, str]


def write_corpus(samples: Sequence[Sample], path: Path) -> None:
    """Persist a corpus as newline-delimited JSON."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fp:
        for prompt, completion in samples:
            payload = {"prompt": prompt, "completion": completion}
            fp.write(json.dumps(payload) + "\n")


_SQUAD_URL = "https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json"


def load_squad_corpus(limit: int | None = None, seed: int = 1037) -> List[Sample]:
    """Load calibration samples from the public SQuAD v2.0 training set.

    Prompts contain the supporting context and question; completions are the
    reference answers. Unanswerable questions are excluded. A random subset is
    returned when `limit` is set.
    """

    try:
        with request.urlopen(_SQUAD_URL) as response:  # pragma: no cover - network fetch
            data = json.loads(response.read().decode("utf-8"))
    except Exception as exc:  # pragma: no cover - network dependency
        raise RuntimeError("Failed to download SQuAD calibration data") from exc

    entries: List[Sample] = []
    for article in data.get("data", []):
        for paragraph in article.get("paragraphs", []):
            context = paragraph.get("context", "").strip()
            if not context:
                continue
            for qa in paragraph.get("qas", []):
                if qa.get("is_impossible"):
                    continue
                answers = qa.get("answers") or []
                if not answers:
                    continue
                answer_text = answers[0].get("text", "").strip()
                if not answer_text:
                    continue
                question = qa.get("question", "").strip()
                if not question:
                    continue
                prompt = f"{context}\n\nQuestion: {question}\nAnswer:"
                entries.append((prompt, answer_text))

    if not entries:
        raise ValueError("No SQuAD samples parsed for calibration")

    if limit is not None and limit < len(entries):
        rng = random.Random(seed)
        rng.shuffle(entries)
        entries = entries[:limit]

    return entries


__all__ = ["write_corpus", "load_squad_corpus"]
