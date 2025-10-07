from __future__ import annotations

from importlib import import_module
from typing import Dict

from .common import Example

_DATASET_MODULES: Dict[str, str] = {
    "truthfulqa": "truthfulqa",
    "mmlu": "mmlu",
    "xsum": "xsum",
    "samsum": "samsum",
    "cnn_dailymail": "cnn_dailymail",
    "wmt14_fren": "wmt14_fren",
    "wmt19_deen": "wmt19_deen",
}


def load_dataset(name: str, config: dict) -> list[Example]:
    if name not in _DATASET_MODULES:
        raise KeyError(f"Unknown dataset {name}")
    module = import_module(f"{__name__}.{_DATASET_MODULES[name]}")
    return module.load(config)
