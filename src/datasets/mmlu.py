from __future__ import annotations

from datasets import load_dataset

from .common import Example, hf_subset


CHOICE_LETTERS = ["A", "B", "C", "D"]


def load(config: dict) -> list[Example]:
    subject = config.get("subject", "abstract_algebra")
    limit = int(config.get("limit", 32))
    split = config.get("split", "validation")
    ds_name = "hendrycksTest"

    def mapper(row: dict) -> Example:
        choices = "\n".join(f"{letter}. {text}" for letter, text in zip(CHOICE_LETTERS, row["choices"]))
        prompt = f"{row['question']}\n{choices}\nAnswer:"
        answer_letter = CHOICE_LETTERS[int(row["answer"])]
        return Example(uid=str(row["idx"]), prompt=prompt, reference=answer_letter, task="qa")

    subset = hf_subset(
        load_dataset,
        ds_name,
        split=f"{subject}-{split}",
        limit=limit,
        mapper=mapper,
    )
    return subset
