#!/usr/bin/env python3
from __future__ import annotations

import json
import argparse
from pathlib import Path
from typing import Dict, Set

from _path import project_root
from src.datasets import load_dataset
from src.uq.io import load_yaml


def _merge_dataset_options(root: Path, ds_cfg: Dict[str, object]) -> Dict[str, object]:
    opts = {k: v for k, v in ds_cfg.items() if k != "name"}
    config_path = opts.pop("config", None)
    if config_path:
        base = load_yaml(root / config_path)
        base.update(opts)
        opts = base
    return opts


def _resolve_local_path(root: Path, opts: Dict[str, object], name: str) -> Path:
    raw = opts.get("local_file")
    if raw is None:
        raw = f"data/static/{name}.jsonl"
    path = Path(raw)
    return path if path.is_absolute() else root / path


def _export_examples(path: Path, examples) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for ex in examples:
            fh.write(json.dumps(ex.__dict__, ensure_ascii=False) + "\n")


def _rel_path(path: Path, root: Path) -> Path:
    try:
        return path.relative_to(root)
    except ValueError:
        return path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Curate small JSONL shards for configured datasets.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/default.yaml"),
        help="Evaluation config to read dataset selections from (default: configs/default.yaml)",
    )
    parser.add_argument(
        "-d",
        "--dataset",
        action="append",
        dest="datasets",
        help="Dataset name to curate (may be provided multiple times). Defaults to all datasets in the config.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    root = project_root()
    config_path = args.config if args.config.is_absolute() else root / args.config
    cfg = load_yaml(config_path)
    datasets_cfg = cfg.get("datasets", [])
    if not datasets_cfg:
        print("No datasets configured; nothing to curate.")
        return
    selected: Set[str] | None = set(args.datasets) if args.datasets else None
    curated: list[str] = []
    for ds_cfg in datasets_cfg:
        name = ds_cfg["name"]
        if selected and name not in selected:
            continue
        opts = _merge_dataset_options(root, ds_cfg)
        local_path = _resolve_local_path(root, opts, name)
        hf_opts = dict(opts)
        hf_opts.pop("local_file", None)
        split = hf_opts.get("split", "validation")
        limit = int(hf_opts.get("limit", 32))
        if isinstance(split, str) and '[:' not in split:
            window = max(limit, limit * 2)
            hf_opts["split"] = f"{split}[:{window}]"
        examples = load_dataset(name, hf_opts)
        _export_examples(local_path, examples)
        rel_path = _rel_path(local_path, root)
        print(f"[curated] {name}: wrote {len(examples)} examples to {rel_path}")
        curated.append(name)
    if selected:
        missing = sorted(selected.difference(curated))
        if missing:
            print(f"[warn] requested datasets not found in config: {', '.join(missing)}")


if __name__ == "__main__":
    main()
