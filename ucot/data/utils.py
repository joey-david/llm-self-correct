"""Utilities for loading lightweight calibration/evaluation data."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Iterator, List, Tuple


Sample = Tuple[str, str]


def load_prompt_completion_pairs(paths: Iterable[Path]) -> List[Sample]:
    """Load `(prompt, reference)` pairs from JSONL or plain text files.

    Expected formats:
    * JSONL with `prompt` and optionally `completion` / `reference` keys
    * Plain text with `prompt\tcompletion`
    """

    samples: List[Sample] = []
    for raw_path in paths:
        path = Path(raw_path)
        if not path.exists():
            raise FileNotFoundError(path)
        if path.suffix == ".jsonl":
            with path.open() as fp:
                for line in fp:
                    record = json.loads(line)
                    prompt = record.get("prompt") or record.get("input")
                    if prompt is None:
                        raise ValueError(f"Missing prompt key in {path}")
                    completion = (
                        record.get("completion")
                        or record.get("reference")
                        or record.get("target")
                        or ""
                    )
                    samples.append((prompt, completion))
        else:
            with path.open() as fp:
                for line in fp:
                    if not line.strip():
                        continue
                    prompt, *rest = line.rstrip("\n").split("\t")
                    completion = rest[0] if rest else ""
                    samples.append((prompt, completion))
    return samples


def batched(iterable: Iterable, batch_size: int) -> Iterator[List]:
    """Yield lists of size `batch_size` from iterable."""

    batch: List = []
    for item in iterable:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch
