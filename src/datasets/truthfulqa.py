from __future__ import annotations

from datasets import load_dataset

from .common import Example, hf_subset


def load(config: dict) -> list[Example]:
    limit = int(config.get("limit", 32))
    split = config.get("split", "validation")
    hf_config = config.get("hf_config", "generation")
    local_file = config.get("local_file")
    return hf_subset(
        load_dataset,
        "truthful_qa",
        split=split,
        name=hf_config,
        limit=limit,
        local_file=local_file,
        mapper=lambda row: Example(
            uid=str(row["question"]),
            prompt=row["question"],
            reference=row["best_answer"],
            task="qa",
        ),
    )
