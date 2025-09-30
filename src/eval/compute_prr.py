#!/usr/bin/env python3
"""Compute dataset-level Prediction Rejection Ratio (PRR) for RAUQ outputs."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

try:  # pragma: no cover - runtime import shim
    from .prr_utils import compute_prr
except ImportError:  # allow running as a standalone script
    from prr_utils import compute_prr  # type: ignore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute PRR for RAUQ runs.")
    parser.add_argument("--infile", type=Path, default=Path("data/artifacts/rauq_output.jsonl"))
    parser.add_argument(
        "--max_fraction",
        type=float,
        default=0.5,
        help="Maximum fraction of instances to reject when computing PRR",
    )
    return parser.parse_args()


def load_records(path: Path) -> List[Dict]:
    records: List[Dict] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def quality_from_record(record: Dict) -> Tuple[float, bool]:
    """Return (quality_score, is_valid)."""
    if record.get("alignscore_best") is not None:
        try:
            val = float(record["alignscore_best"])
            return val, True
        except (TypeError, ValueError):
            pass
    if record.get("correct") is not None:
        return (1.0 if record.get("correct") else 0.0), True
    return 0.0, False


def main() -> None:
    args = parse_args()
    infile = args.infile
    if not infile.is_file():
        raise SystemExit(f"Input file not found: {infile}")

    records = load_records(infile)
    by_dataset: Dict[str, List[Tuple[float, float]]] = defaultdict(list)
    for rec in records:
        dataset = str(rec.get("dataset", "unknown"))
        u = rec.get("u_final")
        quality, ok = quality_from_record(rec)
        if not ok:
            continue
        try:
            uncert = float(u)
        except (TypeError, ValueError):
            continue
        by_dataset[dataset].append((quality, uncert))

    if not by_dataset:
        print("No valid records with both quality metrics and uncertainties.")
        return

    print(f"Loaded {len(records)} records across {len(by_dataset)} datasets.")
    for dataset, pairs in sorted(by_dataset.items()):
        qualities = [q for q, _ in pairs]
        uncertainties = [u for _, u in pairs]
        prr, fractions, curve = compute_prr(qualities, uncertainties, max_fraction=args.max_fraction)
        if np.isnan(prr):
            status = "n/a"
        else:
            status = f"{prr:.3f}"
        print(f"Dataset={dataset:20s} PRR={status}")


if __name__ == "__main__":
    main()
