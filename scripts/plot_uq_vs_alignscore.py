#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from _path import project_root
from src.uq.metrics import alignscore_scores, load_alignscore, plot_scatter


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot RAUQ uncertainty vs AlignScore quality")
    parser.add_argument("--predictions", required=True, help="Path to predictions.jsonl")
    parser.add_argument("--scores", required=True, help="Path to scores_rauq.jsonl")
    parser.add_argument("--output", required=True, help="Output PNG path")
    args = parser.parse_args()
    root = project_root()
    preds_path = Path(args.predictions)
    scores_path = Path(args.scores)
    out_path = Path(args.output)
    if not preds_path.is_absolute():
        preds_path = root / preds_path
    if not scores_path.is_absolute():
        scores_path = root / scores_path
    if not out_path.is_absolute():
        out_path = root / out_path
    preds = [json.loads(line) for line in preds_path.read_text().splitlines() if line.strip()]
    scores = {row["id"]: row["u"] for row in (json.loads(line) for line in scores_path.read_text().splitlines() if line.strip())}
    model = load_alignscore()
    align_scores = alignscore_scores(model, [p["prompt"] for p in preds], [p["prediction"] for p in preds], [p["gold"] for p in preds])
    uq_scores = [scores[p["id"]] for p in preds]
    plot_scatter(uq_scores, align_scores, out_path, "RAUQ (u)", "AlignScore")


if __name__ == "__main__":
    main()
