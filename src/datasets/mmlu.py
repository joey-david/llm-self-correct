from __future__ import annotations

from datasets import load_dataset
from hashlib import sha1

from .common import Example, hf_subset


CHOICE_LETTERS = ["A", "B", "C", "D"]


def load(config: dict) -> list[Example]:
    subject = config.get("subject", "abstract_algebra")
    limit = int(config.get("limit", 32))
    split = config.get("split", "validation")
    ds_name = "cais/mmlu"
    local_file = config.get("local_file")

    def mapper(row: dict) -> Example:
        choices = "\n".join(f"{letter}. {text}" for letter, text in zip(CHOICE_LETTERS, row["choices"]))
        prompt = f"{row['question']}\n{choices}\nAnswer:"
        answer_raw = row["answer"]
        if isinstance(answer_raw, str):
            answer_letter = answer_raw.strip().upper()
            if answer_letter not in CHOICE_LETTERS:
                raise ValueError(f"Unexpected answer label: {answer_raw}")
        else:
            answer_letter = CHOICE_LETTERS[int(answer_raw)]
        subject_tag = row.get("subject", subject)
        raw_id = row.get("id") or row.get("idx")
        if raw_id is None:
            raw_id = sha1(row["question"].encode("utf-8")).hexdigest()[:8]
        uid = f"{subject_tag}::{raw_id}"
        return Example(uid=uid, prompt=prompt, reference=answer_letter, task="qa")

    subset = hf_subset(
        load_dataset,
        ds_name,
        split=split,
        limit=limit,
        name=subject,
        local_file=local_file,
        mapper=mapper,
    )
    return subset
