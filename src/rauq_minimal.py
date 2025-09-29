#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict

import yaml

ROOT = Path(__file__).resolve().parent
for candidate in (ROOT, ROOT.parent):
    if candidate.exists() and str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from rauq_minimal import (  # noqa: E402
    AnswerScorer,
    HeadSelector,
    ModelAdapter,
    PromptBuilder,
    RAUQ,
    Runner,
)
from rauq_minimal.runner import set_seed  # noqa: E402

DEFAULT_CONFIG: Dict[str, Any] = {
    "in": "data/calibration/calibration_10k.jsonl",
    "out": "data/artifacts/rauq_output.jsonl", 
    "model": "Qwen/Qwen3-8B",
    "alpha": 0.2, # as recommended in the paper
    "max_new": 128, # to ensure full answers
    "device": "auto",
    "dtype": "auto",
    "store_all_heads": False,
    "seed": 42, # the normie seed
    "dataset_fraction": 0.1,
}
REQUIRED_KEYS = ("in", "out")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal RAUQ runner")
    parser.add_argument("--config", help="Path to YAML config file")
    args = parser.parse_args()
    if args.config is None:
        # Use default config if no config file is provided
        args.config = None
    return args

def load_config(config_path: Path) -> Dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as fh:
        loaded = yaml.safe_load(fh) or {}

    if not isinstance(loaded, dict):
        raise ValueError("Config file must contain a mapping at the top level")

    config: Dict[str, Any] = DEFAULT_CONFIG.copy()
    config.update(loaded)

    fraction = config.get("dataset_fraction", DEFAULT_CONFIG["dataset_fraction"])
    if fraction is None:
        fraction = DEFAULT_CONFIG["dataset_fraction"]
    try:
        fraction_val = float(fraction)
    except (TypeError, ValueError) as exc:
        raise ValueError("dataset_fraction must be a float between 0 and 1") from exc
    if fraction_val < 0.0 or fraction_val > 1.0:
        raise ValueError("dataset_fraction must be between 0 and 1")
    config["dataset_fraction"] = fraction_val

    missing = [key for key in REQUIRED_KEYS if not config.get(key)]
    if missing:
        raise ValueError(f"Config missing required field(s): {', '.join(missing)}")

    return config


def main() -> None:
    args = parse_args()
    if args.config:
        config = load_config(Path(args.config))
    else:
        # Use default config when no config file is provided
        config = DEFAULT_CONFIG.copy()
    set_seed(int(config.get("seed", DEFAULT_CONFIG["seed"])))

    model_adapter = ModelAdapter(
        model_name=str(config.get("model", DEFAULT_CONFIG["model"])),
        device=str(config.get("device", DEFAULT_CONFIG["device"])),
        dtype=config.get("dtype", DEFAULT_CONFIG["dtype"]),
    )
    prompt_builder = PromptBuilder()
    scorer = AnswerScorer()
    head_selector = HeadSelector()
    rauq = RAUQ(alpha=float(config.get("alpha", DEFAULT_CONFIG["alpha"])))

    runner = Runner(
        model_adapter=model_adapter,
        prompt_builder=prompt_builder,
        scorer=scorer,
        head_selector=head_selector,
        rauq=rauq,
        max_new_tokens=int(config.get("max_new", DEFAULT_CONFIG["max_new"])),
        store_all_heads=bool(config.get("store_all_heads", DEFAULT_CONFIG["store_all_heads"])),
        dataset_fraction=float(config.get("dataset_fraction", DEFAULT_CONFIG["dataset_fraction"])),
    )

    runner.run(str(config["in"]), str(config["out"]))


if __name__ == "__main__":
    main()
