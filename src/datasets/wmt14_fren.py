from __future__ import annotations

from datasets import load_dataset

from .common import Example, hf_subset


def load(config: dict) -> list[Example]:
    limit = int(config.get("limit", 32))
    split = config.get("split", "validation")
    return hf_subset(
        load_dataset,
        "wmt14",
        split=split,
        config_name="fr-en",
        limit=limit,
        mapper=lambda row: Example(
            uid=str(row["id"]),
            prompt=f"Translate the following French sentence to English:\n\n{row['translation']['fr']}",
            reference=row["translation"]["en"],
            task="mt",
        ),
    )
