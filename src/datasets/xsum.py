from __future__ import annotations

from functools import lru_cache
from hashlib import sha1

from datasets import load_dataset
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import EntryNotFoundError

from .common import Example, hf_subset


def load(config: dict) -> list[Example]:
    limit = int(config.get("limit", 16))
    split = config.get("split", "validation")
    local_file = config.get("local_file")
    ds_name = config.get("hf_dataset", "GEM/xsum")
    document_key = config.get("document_key", "document")
    summary_key = config.get("summary_key", "summary")
    parquet_config = config.get("parquet_fallback")

    def _resolve_text(row: dict) -> tuple[str, str]:
        document = (
            row.get(document_key)
            or row.get("document")
            or row.get("article")
            or row.get("source")
            or row.get("text")
        )
        summary = (
            row.get(summary_key)
            or row.get("summary")
            or row.get("target")
            or row.get("highlights")
        )
        if summary is None and isinstance(row.get("references"), list):
            summary = row["references"][0]
        if document is None or summary is None:
            raise KeyError("Missing document or summary field in XSum sample")
        return document, summary

    def mapper(row: dict) -> Example:
        document, summary = _resolve_text(row)
        raw_id = row.get("id") or row.get("gem_id")
        if raw_id is None:
            raw_id = sha1(document.encode("utf-8")).hexdigest()[:8]
        prompt = f"Summarize the following article:\n\n{document}"
        return Example(uid=str(raw_id), prompt=prompt, reference=summary, task="summ")

    def _fallback(split_name: str):
        if parquet_config is not None:
            return parquet_config.get(split_name)
        return _xsum_parquet_path(split_name)

    return hf_subset(
        load_dataset,
        ds_name,
        split=split,
        limit=limit,
        local_file=local_file,
        parquet_fallback=_fallback,  # callable: returns a local file path
        mapper=mapper,
    )


# Paths for GEM repo layout
_PARQUET_FILENAMES = {
    "train": "xsum/gem-train.parquet",
    "validation": "xsum/gem-validation.parquet",
    "test": "xsum/gem-test.parquet",
}

# Known commits where the XSum parquet files exist in GEM/gem
_GEM_COMMITS_WITH_XSUM = [
    # all 3 xsum parquet files are present across these
    "408cee5176bceec74fb6ac7c2f134661836e8b14",
    "c4ec4d199f645927ac42870f184f292e9536e138",
    "93abb4cb8c5f67da14356845ecf28318a0e8e7c4",
]


@lru_cache(maxsize=None)
def _xsum_parquet_path(split_name: str) -> str | None:
    """Return a LOCAL file path to the parquet for the given split, trying:
    1) GEM/gem at main
    2) GEM/gem pinned commits (above)
    3) EdinburghNLP/xsum at refs/convert/parquet
    """
    if split_name not in _PARQUET_FILENAMES:
        return None

    # 1) Try GEM/gem @ main
    try:
        return hf_hub_download(
            repo_id="GEM/gem",
            repo_type="dataset",
            filename=_PARQUET_FILENAMES[split_name],
        )
    except EntryNotFoundError:
        pass

    # 2) Try pinned commits where files exist
    for rev in _GEM_COMMITS_WITH_XSUM:
        try:
            return hf_hub_download(
                repo_id="GEM/gem",
                repo_type="dataset",
                filename=_PARQUET_FILENAMES[split_name],
                revision=rev,
            )
        except EntryNotFoundError:
            continue

    # 3) Fallback to canonical XSum parquet conversion by EdinburghNLP
    # Note different layout: default/{split}/0000.parquet
    ed_file = f"default/{split_name}/0000.parquet"
    return hf_hub_download(
        repo_id="EdinburghNLP/xsum",
        repo_type="dataset",
        filename=ed_file,
        revision="refs/convert/parquet",
    )
