from __future__ import annotations

from datasets import load_dataset

from .common import Example, hf_subset


def load(config: dict) -> list[Example]:
    limit = int(config.get("limit", 32))
    split = config.get("split", "validation")
    hf_config = config.get("hf_config", "generation")
    return hf_subset(
        load_dataset,
        "truthful_qa",
        split=split,
        config_name=hf_config,
        limit=limit,
        mapper=lambda row: Example(
            uid=str(row["question"]),
            prompt=row["question"],
            reference=row["best_answer"],
            task="qa",
        ),
    )
