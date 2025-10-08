from __future__ import annotations
from functools import lru_cache
from hashlib import sha1

from datasets import load_dataset
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import EntryNotFoundError

from .common import Example, hf_subset

def load(config: dict) -> list[Example]:
    limit = int(config.get("limit", 8))
    split = config.get("split", "validation")
    local_file = config.get("local_file")
    # keep your existing ID; script load will fail → fallback kicks in
    ds_name = config.get("hf_dataset", "ccdv/cnn_dailymail")
    parquet_config = config.get("parquet_fallback")
    version = str(config.get("hf_config", "3.0.0"))  # prefer 3.0.0; we’ll fallback if missing
    document_key = config.get("document_key", "article")
    summary_key  = config.get("summary_key", "highlights")

    def mapper(row: dict) -> Example:
        doc = row.get(document_key) or row.get("document") or row.get("text")
        summ = row.get(summary_key) or row.get("summary") or row.get("target")
        if doc is None or summ is None:
            raise KeyError("Missing article/highlights in CNN/DailyMail sample")
        rid = row.get("id") or sha1(doc.encode("utf-8")).hexdigest()[:8]
        prompt = f"Summarize the following article:\n\n{doc}"
        return Example(uid=str(rid), prompt=prompt, reference=summ, task="summ")

    def _fallback(split_name: str) -> str | None:
        if parquet_config is not None:
            return parquet_config.get(split_name)
        return _cnn_parquet_path(split_name, version)

    return hf_subset(
        load_dataset,
        ds_name,
        split=split,
        limit=limit,
        local_file=local_file,
        parquet_fallback=_fallback,  # returns a local parquet path
        mapper=mapper,
        name=version,  # harmless when script path fails
    )

# --- Parquet fallback implementation ---

_SPLIT_FILE = {
    "train": "train-00000-of-00001.parquet",
    "validation": "validation-00000-of-00001.parquet",
    "test": "test-00000-of-00001.parquet",
}

# versions that ship parquet in abisee/cnn_dailymail
_KNOWN_VERSIONS = ("3.0.0", "2.0.0", "1.0.0")

@lru_cache(maxsize=None)
def _cnn_parquet_path(split_name: str, preferred_version: str = "3.0.0") -> str | None:
    if split_name not in _SPLIT_FILE:
        return None

    # Try preferred version first, then fall back to older ones.
    versions = (preferred_version,) + tuple(v for v in _KNOWN_VERSIONS if v != preferred_version)
    for ver in versions:
        try:
            return hf_hub_download(
                repo_id="abisee/cnn_dailymail",
                repo_type="dataset",
                filename=f"{ver}/{_SPLIT_FILE[split_name]}",
            )
        except EntryNotFoundError:
            continue

    # Last-ditch: try CCDV mirror if it ever gets parquet shards
    for ver in versions:
        try:
            return hf_hub_download(
                repo_id="ccdv/cnn_dailymail",
                repo_type="dataset",
                filename=f"{ver}/{_SPLIT_FILE[split_name]}",
            )
        except EntryNotFoundError:
            continue

    raise FileNotFoundError("Could not locate CNN/DailyMail parquet shard in known locations.")
