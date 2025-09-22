#!/usr/bin/env python
"""CLI for fitting the RAUQ trigger threshold."""
from __future__ import annotations

import argparse
from pathlib import Path

from ucot.config import ThresholdTrainingConfig
from ucot.threshold import exact_match_metric, train_threshold
from ucot.utils.logging import setup_logging


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fit logistic threshold for RAUQ triggers")
    parser.add_argument("--model", required=True, help="Model name or path")
    parser.add_argument("--tokenizer", help="Tokenizer name (defaults to model)")
    parser.add_argument("--calibration", nargs="+", type=Path, required=True)
    parser.add_argument("--heads", type=Path, default=Path("artifacts/head_selection.json"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/threshold.json"))
    parser.add_argument("--alpha", type=float, default=0.3)
    parser.add_argument("--max-samples", type=int, default=2048)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--penalty", type=float, default=1.0, help="Inverse regularisation strength (C)")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    setup_logging()

    config = ThresholdTrainingConfig(
        calibration_paths=list(args.calibration),
        model_name=args.model,
        tokenizer_name=args.tokenizer,
        head_indices_path=args.heads,
        output_path=args.output,
        alpha=args.alpha,
        max_samples=args.max_samples,
        device=args.device,
        logistic_penalty=args.penalty,
    )
    train_threshold(config, metric_fn=exact_match_metric)


if __name__ == "__main__":
    main()
