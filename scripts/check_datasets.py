#!/usr/bin/env python3
from __future__ import annotations

import json
import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from _path import project_root
from src.uq.io import load_yaml


def _merge_options(root: Path, ds_cfg: Dict[str, Any]) -> Dict[str, Any]:
    opts = {k: v for k, v in ds_cfg.items() if k != "name"}
    config_path = opts.pop("config", None)
    if config_path:
        base = load_yaml(root / config_path)
        base.update(opts)
        opts = base
    return opts


def _resolve_local_path(root: Path, opts: Dict[str, Any], name: str) -> Path | None:
    raw = opts.get("local_file")
    if raw is None:
        return None
    path = Path(raw)
    return path if path.is_absolute() else root / path


def _preview_record(path: Path) -> Dict[str, Any] | None:
    try:
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    return json.loads(line)
    except FileNotFoundError:
        return None
    return None


def _summarise_opts(opts: Dict[str, Any]) -> Dict[str, Any]:
    keys = ["split", "limit", "subject", "hf_config"]
    return {k: opts[k] for k in keys if k in opts}


def _rel_path(path: Path, root: Path) -> Path:
    try:
        return path.relative_to(root)
    except ValueError:
        return path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify curated dataset shards exist.")
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
        help="Dataset name to check (may be provided multiple times). Defaults to all datasets in the config.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    root = project_root()
    config_path = args.config if args.config.is_absolute() else root / args.config
    cfg = load_yaml(config_path)
    datasets_cfg = cfg.get("datasets", [])
    selected: Optional[Set[str]] = set(args.datasets) if args.datasets else None
    missing: List[str] = []
    for ds_cfg in datasets_cfg:
        name = ds_cfg["name"]
        if selected and name not in selected:
            continue
        opts = _merge_options(root, ds_cfg)
        local_path = _resolve_local_path(root, opts, name)
        meta = _summarise_opts(opts)
        if local_path is None:
            print(f"[WARN] {name}: no local_file configured; meta={meta}")
            continue
        if local_path.exists():
            preview = _preview_record(local_path)
            sample_id = preview.get("uid") if preview else None
            rel = _rel_path(local_path, root)
            print(f"[OK] {name}: meta={meta} file={rel} sample_id={sample_id}")
        else:
            missing.append(name)
            rel = _rel_path(local_path, root)
            print(f"[MISS] {name}: expected curated file {rel} (meta={meta})")
    if selected:
        missing_cfg = sorted(set(args.datasets) - {ds_cfg["name"] for ds_cfg in datasets_cfg})
        if missing_cfg:
            print(f"[WARN] datasets not present in config: {', '.join(missing_cfg)}")
    if missing:
        joined = ", ".join(missing)
        raise SystemExit(f"missing curated datasets: {joined}")
    print("All configured datasets have curated files.")


if __name__ == "__main__":
    main()
