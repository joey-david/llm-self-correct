#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import yaml

from _path import project_root
from src.uq.eval import run_eval


def main() -> None:
    parser = argparse.ArgumentParser(description="Run RAUQ evaluation")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--output-dir")
    parser.add_argument("--force-cpu", action="store_true")
    args = parser.parse_args()
    root = project_root()
    cfg = Path(args.config)
    if not cfg.is_absolute():
        cfg = root / cfg
    data = yaml.safe_load(cfg.read_text())
    if args.force_cpu:
        data["force_cpu"] = True
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as tmp:
        yaml.safe_dump(data, tmp)
        tmp_path = Path(tmp.name)
    output_dir = Path(args.output_dir) if args.output_dir else None
    if output_dir and not output_dir.is_absolute():
        output_dir = root / output_dir
    result = run_eval(tmp_path, output_dir)
    print(json.dumps(result.metrics, indent=2))


if __name__ == "__main__":
    main()
