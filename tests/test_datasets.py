from __future__ import annotations

import pytest

from src.datasets import load_dataset
from src.datasets.smoke import SMOKE_CONFIGS


@pytest.mark.parametrize("name,cfg", SMOKE_CONFIGS.items())
def test_dataset_loader_smoke(name: str, cfg: dict) -> None:
    config = dict(cfg)
    # ensure deterministic tiny sample size
    config["limit"] = min(int(config.get("limit", 1)), 2)
    records = load_dataset(name, config)
    assert records, f"{name} loader returned no examples"
    first = records[0]
    assert first.prompt, f"{name} example missing prompt"
    assert first.reference is not None, f"{name} example missing reference"
