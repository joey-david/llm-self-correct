from __future__ import annotations

from datasets import load_dataset

from .common import Example, hf_subset


def load(config: dict) -> list[Example]:
    limit = int(config.get("limit", 16))
    split = config.get("split", "validation")
    return hf_subset(
        load_dataset,
        "samsum",
        split=split,
        limit=limit,
        mapper=lambda row: Example(
            uid=row["id"],
            prompt=f"Summarize the dialogue:\n\n{row['dialogue']}",
            reference=row["summary"],
            task="summ",
        ),
    )
