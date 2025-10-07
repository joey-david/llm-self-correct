from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable, Dict, List

from ..uq.io import GLOBAL_SEED


@dataclass
class Example:
    uid: str
    prompt: str
    reference: str
    task: str


def _rng(seed: int | None = None) -> random.Random:
    return random.Random(GLOBAL_SEED if seed is None else seed)


def hf_subset(
    loader: Callable[..., object],
    dataset_name: str,
    split: str,
    limit: int,
    mapper: Callable[[Dict[str, object]], Example],
    seed: int | None = None,
    **kwargs: object,
) -> List[Example]:
    try:
        ds = loader(dataset_name, split=split, **kwargs)  # type: ignore[arg-type]
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(f"Failed to load dataset {dataset_name}: {exc}")
    indices = list(range(len(ds)))
    _rng(seed).shuffle(indices)
    selected = indices[:limit]
    return [mapper(ds[i]) for i in selected]
