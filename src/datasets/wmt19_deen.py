from __future__ import annotations

from datasets import load_dataset

from .common import Example, hf_subset


def load(config: dict) -> list[Example]:
    limit = int(config.get("limit", 32))
    split = config.get("split", "validation")
    local_file = config.get("local_file")
    ds_name = config.get("hf_dataset", "wmt/wmt19")
    hf_config = config.get("hf_config", "de-en")
    parquet_config = config.get("parquet_fallback")

    def mapper(row: dict) -> Example:
        translation = row.get("translation") or {}
        src_text = translation.get("de") or row.get("de")
        tgt_text = translation.get("en") or row.get("en")
        if src_text is None or tgt_text is None:
            raise KeyError("Missing de-en translation fields in WMT19 sample")
        prompt = f"Translate the following German sentence to English:\n\n{src_text}"
        return Example(uid="", prompt=prompt, reference=tgt_text, task="mt")

    def _fallback(split_name: str) -> str | None:
        if parquet_config is not None:
            return parquet_config.get(split_name)
        if hf_config == "de-en" and split_name == "validation":
            return "https://huggingface.co/datasets/wmt/wmt19/resolve/main/de-en/validation-00000-of-00001.parquet"
        return None

    return hf_subset(
        load_dataset,
        ds_name,
        split=split,
        config_name=hf_config,
        limit=limit,
        local_file=local_file,
        parquet_fallback=_fallback,
        mapper=mapper,
    )
