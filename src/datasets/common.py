from __future__ import annotations

import random
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List

from datasets import load_dataset as _hf_load_dataset

from ..uq.io import GLOBAL_SEED


@dataclass
class Example:
    uid: str
    prompt: str
    reference: str
    task: str


def _rng(seed: int | None = None) -> random.Random:
    return random.Random(GLOBAL_SEED if seed is None else seed)


def _load_local_examples(path: Path) -> List[Example]:
    with path.open("r", encoding="utf-8") as fh:
        return [Example(**json.loads(line)) for line in fh if line.strip()]


def _select_indices(total: int, limit: int, seed: int | None) -> List[int]:
    if limit <= 0 or total <= 0:
        return []
    limit = min(limit, total)
    indices = list(range(total))
    _rng(seed).shuffle(indices)
    return indices[:limit]


def hf_subset(
    loader: Callable[..., object],
    dataset_name: str,
    split: str,
    limit: int,
    mapper: Callable[[Dict[str, object]], Example],
    seed: int | None = None,
    local_file: str | Path | None = None,
    parquet_fallback: Dict[str, object] | Callable[[str], object] | None = None,
    **kwargs: object,
) -> List[Example]:
    if local_file is not None:
        path = Path(local_file)
        if path.exists():
            local_examples = _load_local_examples(path)
            selected = _select_indices(len(local_examples), limit, seed)
            return [local_examples[idx] for idx in selected]
    try:
        ds = loader(dataset_name, split=split, **kwargs)  # type: ignore[arg-type]
    except Exception as exc:  # pragma: no cover
        base_split = split.split("[", 1)[0]
        if callable(parquet_fallback):
            fallback_files = parquet_fallback(base_split)
        elif parquet_fallback is not None:
            fallback_files = parquet_fallback.get(base_split)
        else:
            fallback_files = None
        if fallback_files is None:
            raise RuntimeError(f"Failed to load dataset {dataset_name}: {exc}")
        try:
            data_files = {base_split: fallback_files}
            ds = _hf_load_dataset("parquet", data_files=data_files, split=split)
        except Exception as exc_parquet:  # pragma: no cover
            raise RuntimeError(
                f"Failed to load dataset {dataset_name} (script error: {exc}); parquet fallback error: {exc_parquet}"
            )
    selected = _select_indices(len(ds), limit, seed)
    return [mapper(ds[i]) for i in selected]
