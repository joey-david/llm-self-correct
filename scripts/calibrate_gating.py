#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from _path import project_root
from src.aspects import UtilityWeights, calibrate_configs


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate gating thresholds against utility")
    parser.add_argument("--grid", required=True, help="YAML file with candidate configs")
    parser.add_argument("--stats", required=True, help="JSON file with calibration statistics")
    parser.add_argument("--value", type=float, default=1.0)
    parser.add_argument("--cost-cot", type=float, default=0.1)
    parser.add_argument("--cost-rb", type=float, default=0.01)
    parser.add_argument("--cost-latency", type=float, default=0.001)
    args = parser.parse_args()
    root = project_root()
    grid_path = Path(args.grid)
    stats_path = Path(args.stats)
    if not grid_path.is_absolute():
        grid_path = root / grid_path
    if not stats_path.is_absolute():
        stats_path = root / stats_path
    grid = yaml.safe_load(grid_path.read_text())
    stats = json.loads(stats_path.read_text())
    weights = UtilityWeights(
        value=args.value,
        cost_cot=args.cost_cot,
        cost_rb=args.cost_rb,
        cost_latency=args.cost_latency,
    )
    best_cfg, utility = calibrate_configs(grid, stats, weights)
    print(json.dumps({"best_config": best_cfg, "utility": utility}, indent=2))


if __name__ == "__main__":
    main()
