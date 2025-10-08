from __future__ import annotations

from typing import Dict

SMOKE_CONFIGS: Dict[str, dict] = {
    "truthfulqa": {"limit": 1, "split": "validation", "hf_config": "generation"},
    "mmlu": {"limit": 1, "split": "validation", "subject": "abstract_algebra"},
    "xsum": {"limit": 1, "split": "validation"},
    "samsum": {"limit": 1, "split": "validation"},
    "cnn_dailymail": {"limit": 1, "split": "validation"},
    "wmt14_fren": {"limit": 1, "split": "validation"},
    "wmt19_deen": {"limit": 1, "split": "validation"},
}


def smoke_configs() -> Dict[str, dict]:
    """Return default smoke-test configs for each dataset."""
    return SMOKE_CONFIGS.copy()
