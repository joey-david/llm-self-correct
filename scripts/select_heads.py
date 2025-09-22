#!/usr/bin/env python
"""CLI for selecting RAUQ uncertainty-aware heads."""
from __future__ import annotations

import argparse
from pathlib import Path

from ucot.config import HeadSelectionConfig
from ucot.head_selection import select_uncertainty_heads
from ucot.utils.logging import setup_logging


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Select uncertainty-aware attention heads for RAUQ")
    parser.add_argument("--model", required=True, help="Model name or path")
    parser.add_argument("--tokenizer", help="Tokenizer name (defaults to model)")
    parser.add_argument("--calibration", nargs="+", type=Path, help="Calibration data files (JSONL or TSV)")
    parser.add_argument("--output", type=Path, default=Path("artifacts/head_selection.json"))
    parser.add_argument("--num-examples", type=int, default=256)
    parser.add_argument("--layers-fraction", type=float, default=0.33)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-progress", action="store_true", help="Disable tqdm progress bar")
    parser.add_argument("--no-head-stats", action="store_true", help="Disable per-layer head statistics logging")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    setup_logging()

    config = HeadSelectionConfig(
        calibration_paths=list(args.calibration),
        model_name=args.model,
        tokenizer_name=args.tokenizer,
        output_path=args.output,
        num_examples=args.num_examples,
        layers_fraction=args.layers_fraction,
        device=args.device,
        show_progress=not args.no_progress,
        log_head_stats=not args.no_head_stats,
    )
    select_uncertainty_heads(config)


if __name__ == "__main__":
    main()
