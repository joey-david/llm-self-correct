from __future__ import annotations

from functools import lru_cache
from hashlib import sha1

from datasets import load_dataset
from huggingface_hub import hf_hub_download

from .common import Example, hf_subset


def load(config: dict) -> list[Example]:
    limit = int(config.get("limit", 16))
    split = config.get("split", "validation")
    local_file = config.get("local_file")
    ds_name = config.get("hf_dataset", "knkarthick/samsum")
    dialogue_key = config.get("document_key", "dialogue")
    summary_key = config.get("summary_key", "summary")
    parquet_config = config.get("parquet_fallback")

    def mapper(row: dict) -> Example:
        dialogue = row.get(dialogue_key) or row.get("dialogue") or row.get("source") or row.get("text")
        summary = row.get(summary_key) or row.get("summary") or row.get("target")
        if summary is None and isinstance(row.get("references"), list):
            summary = row["references"][0]
        if dialogue is None or summary is None:
            raise KeyError("Missing dialogue or summary field in SAMSum sample")
        raw_id = row.get("id") or row.get("gem_id") or sha1(dialogue.encode("utf-8")).hexdigest()[:8]
        prompt = f"Summarize the dialogue:\n\n{dialogue}"
        return Example(uid=str(raw_id), prompt=prompt, reference=summary, task="summ")

    def _fallback(split_name: str) -> str | None:
        if parquet_config is not None:
            return parquet_config.get(split_name)
        return _samsum_parquet_path(split_name)

    return hf_subset(
        load_dataset,
        ds_name,
        split=split,
        limit=limit,
        local_file=local_file,
        parquet_fallback=_fallback,
        mapper=mapper,
    )


_PARQUET_FILES = {
    "train": "default/train/0000.parquet",
    "validation": "default/validation/0000.parquet",
    "test": "default/test/0000.parquet",
}


@lru_cache(maxsize=None)
def _samsum_parquet_path(split_name: str) -> str | None:
    if split_name not in _PARQUET_FILES:
        return None
    return hf_hub_download(
        repo_id="knkarthick/samsum",
        repo_type="dataset",
        filename=_PARQUET_FILES[split_name],
        revision="refs/convert/parquet",
    )
