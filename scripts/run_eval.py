#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import yaml

from src.uq.eval import run_eval


def main() -> None:
    parser = argparse.ArgumentParser(description="Run RAUQ evaluation")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--output-dir")
    parser.add_argument("--force-cpu", action="store_true")
    args = parser.parse_args()
    cfg = Path(args.config)
    data = yaml.safe_load(cfg.read_text())
    if args.force_cpu:
        data["force_cpu"] = True
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as tmp:
        yaml.safe_dump(data, tmp)
        tmp_path = Path(tmp.name)
    result = run_eval(tmp_path, Path(args.output_dir) if args.output_dir else None)
    print(json.dumps(result.metrics, indent=2))


if __name__ == "__main__":
    main()
