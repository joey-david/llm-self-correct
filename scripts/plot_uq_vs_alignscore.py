#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.uq.metrics import alignscore_scores, load_alignscore, plot_scatter


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot RAUQ uncertainty vs AlignScore quality")
    parser.add_argument("--predictions", required=True, help="Path to predictions.jsonl")
    parser.add_argument("--scores", required=True, help="Path to scores_rauq.jsonl")
    parser.add_argument("--output", required=True, help="Output PNG path")
    args = parser.parse_args()
    preds = [json.loads(line) for line in Path(args.predictions).read_text().splitlines() if line.strip()]
    scores = {row["id"]: row["u"] for row in (json.loads(line) for line in Path(args.scores).read_text().splitlines() if line.strip())}
    model = load_alignscore()
    align_scores = alignscore_scores(model, [p["prompt"] for p in preds], [p["prediction"] for p in preds], [p["gold"] for p in preds])
    uq_scores = [scores[p["id"]] for p in preds]
    plot_scatter(uq_scores, align_scores, Path(args.output), "RAUQ (u)", "AlignScore")


if __name__ == "__main__":
    main()
