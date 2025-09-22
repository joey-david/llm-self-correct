#!/usr/bin/env python
"""Run scripted ablations for RAUQ-triggered decoding."""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from statistics import mean
from typing import Dict, List, Optional

import numpy as np
import torch

from ucot.config import ControllerConfig, RAUQConfig
from ucot.controller import RAUQController
from ucot.data.benchmarks import load_benchmark
from ucot.experiments.metrics import METRICS, exact_match
from ucot.rauq import RAUQScorer
from ucot.threshold import ThresholdResult
from ucot.uncertainty import (
    EntropyScorer,
    LogitMarginScorer,
    RAUQScorerWrapper,
    TokenUncertaintyScorer,
)
from ucot.utils.logging import setup_logging
from ucot.utils.model import load_model

BENCHMARK_CHOICES = ["gsm8k", "math", "humaneval"]
TRIGGER_CHOICES = ["rauq", "entropy", "margin"]
REPAIR_CHOICES = ["cot", "rerank", "none"]
COT_POLICIES = ["rauq", "max20", "unlimited"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ablation sweeps for trigger/repair policies")
    parser.add_argument("--model", required=True, help="Model name or path")
    parser.add_argument("--tokenizer", help="Tokenizer name (defaults to model)")
    parser.add_argument("--benchmark", choices=BENCHMARK_CHOICES, required=True)
    parser.add_argument("--split", help="Dataset split override (defaults to canonical test split)")
    parser.add_argument("--limit", type=int, help="Limit number of samples")
    parser.add_argument("--device", default="cuda")

    parser.add_argument("--trigger", choices=TRIGGER_CHOICES, default="rauq")
    parser.add_argument("--alpha", type=float, default=0.3, help="Alpha mix for RAUQ (ignored for entropy/margin)")

    parser.add_argument("--threshold-type", choices=["learned", "fixed"], default="learned")
    parser.add_argument("--threshold", type=Path, default=Path("artifacts/threshold.json"))
    parser.add_argument("--theta", type=float, help="Manual threshold when using --threshold-type fixed")

    parser.add_argument("--heads", type=Path, default=Path("artifacts/head_selection.json"))

    parser.add_argument("--repair", choices=REPAIR_CHOICES, default="cot")
    parser.add_argument("--rollback", type=int, default=2)
    parser.add_argument("--rollback-mode", choices=["fixed", "anchor"], default="fixed")
    parser.add_argument("--cooldown", type=int, default=5)
    parser.add_argument("--stability-window", type=int, default=2)
    parser.add_argument("--max-triggers", type=int, default=5)

    parser.add_argument("--cot-policy", choices=COT_POLICIES, default="rauq")
    parser.add_argument("--cot-length", type=int, help="Override CoT length cap")
    parser.add_argument("--cot-candidates", type=int, default=3)
    parser.add_argument("--cot-temperature", type=float, default=0.7)
    parser.add_argument("--cot-top-p", type=float, default=0.95)

    parser.add_argument("--lookahead", type=int, default=4, help="Lookahead horizon for scoring continuations")

    parser.add_argument("--rerank-candidates", type=int, default=3)
    parser.add_argument("--rerank-temperature", type=float, default=0.7)
    parser.add_argument("--rerank-top-p", type=float, default=0.95)
    parser.add_argument("--rerank-horizon", type=int, default=4)

    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, help="Optional JSON dump of metrics and run details")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_scorer(
    trigger: str,
    alpha: float,
    heads_path: Path,
    device: str,
    num_layers: int,
) -> TokenUncertaintyScorer:
    if trigger == "rauq":
        if not heads_path.exists():
            raise FileNotFoundError(f"Head selection file not found: {heads_path}")
        rauq_config = RAUQConfig(alpha=alpha, head_indices_path=heads_path, device=device)
        base = RAUQScorer.from_config(rauq_config, num_layers=num_layers)
        return RAUQScorerWrapper(
            alpha=base.alpha,
            head_indices=base.head_indices,
            layers=base.layers,
            eps=base.eps,
            device=base.device,
        )
    if trigger == "entropy":
        return EntropyScorer()
    if trigger == "margin":
        return LogitMarginScorer()
    raise ValueError(f"Unsupported trigger: {trigger}")


def configure_cot(config: ControllerConfig, policy: str, cot_length_override: Optional[int]) -> None:
    if policy == "rauq":
        config.cot.stop_mode = "rauq"
        config.cot.max_cot_tokens = cot_length_override or config.cot.max_cot_tokens
    elif policy == "max20":
        config.cot.stop_mode = "fixed"
        config.cot.max_cot_tokens = cot_length_override or 20
    elif policy == "unlimited":
        config.cot.stop_mode = "none"
        config.cot.max_cot_tokens = cot_length_override or 200
    else:
        raise ValueError(f"Unknown CoT policy: {policy}")
    config.cot.cot_prefix = (
        f"Wait, let's quickly think step by step about this (<{config.cot.max_cot_tokens} tokens)."
    )


def main() -> None:
    args = parse_args()
    logger = setup_logging()
    set_seed(args.seed)

    logger.info("Loading benchmark %s", args.benchmark)
    samples = load_benchmark(args.benchmark, split=args.split, limit=args.limit)
    if not samples:
        raise ValueError("No samples loaded for benchmark")

    metric_fn = METRICS.get(args.benchmark, exact_match)

    logger.info("Loading model %s", args.model)
    loaded = load_model(
        model_name=args.model,
        tokenizer_name=args.tokenizer,
        device=args.device,
    )

    scorer = build_scorer(
        trigger=args.trigger,
        alpha=args.alpha,
        heads_path=args.heads,
        device=args.device,
        num_layers=loaded.model.config.num_hidden_layers,
    )

    if args.threshold_type == "learned":
        if not args.threshold.exists():
            raise FileNotFoundError(f"Threshold file not found: {args.threshold}")
        threshold = ThresholdResult.load(args.threshold)
        theta = threshold.theta
        logger.info("Loaded threshold θ=%.4f from %s", theta, args.threshold)
    else:
        if args.theta is None:
            raise ValueError("--theta must be provided when --threshold-type=fixed")
        threshold = None
        theta = args.theta
        logger.info("Using fixed threshold θ=%.4f", theta)

    controller_config = ControllerConfig(
        model_name=args.model,
        tokenizer_name=args.tokenizer,
        theta=theta,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        alpha=args.alpha,
    )
    controller_config.repair_strategy = args.repair
    controller_config.rollback.rollback_depth = args.rollback
    controller_config.rollback.mode = args.rollback_mode
    controller_config.rollback.cooldown = args.cooldown
    controller_config.rollback.stability_window = args.stability_window
    controller_config.rollback.max_triggers = args.max_triggers

    controller_config.cot.candidates = args.cot_candidates
    controller_config.cot.temperature = args.cot_temperature
    controller_config.cot.top_p = args.cot_top_p
    controller_config.cot.lookahead_horizon = args.lookahead
    configure_cot(controller_config, args.cot_policy, args.cot_length)

    controller_config.rerank.candidates = args.rerank_candidates
    controller_config.rerank.temperature = args.rerank_temperature
    controller_config.rerank.top_p = args.rerank_top_p
    controller_config.rerank.lookahead_horizon = args.rerank_horizon

    controller = RAUQController(
        loaded=loaded,
        config=controller_config,
        scorer=scorer,
        threshold=threshold,
    )

    logger.info(
        "Starting evaluation: %s | trigger=%s | repair=%s | samples=%d",
        args.benchmark,
        args.trigger,
        args.repair,
        len(samples),
    )

    results: List[Dict[str, float]] = []
    latencies: List[float] = []

    for sample in samples:
        start = time.perf_counter()
        output = controller.generate(sample.prompt)
        latency = time.perf_counter() - start
        latencies.append(latency)

        is_correct = metric_fn({"reference": sample.reference}, output.completion)
        record = {
            "id": sample.metadata.get("id")
            or sample.metadata.get("task_id")
            or sample.metadata.get("level"),
            "correct": bool(is_correct),
            "triggers": len(output.trigger_events),
            "generated_tokens": len(output.completion_tokens),
            "total_tokens": output.total_tokens,
            "extra_tokens": output.extra_tokens,
            "latency_sec": latency,
        }
        results.append(record)

    performance = mean(r["correct"] for r in results)
    avg_generated = mean(r["generated_tokens"] for r in results)
    avg_total = mean(r["total_tokens"] for r in results)
    performance_per_token = performance / avg_generated if avg_generated else 0.0
    avg_triggers = mean(r["triggers"] for r in results)
    tokens_total = sum(r["generated_tokens"] for r in results)
    correct_total = sum(r["correct"] for r in results)
    tokens_per_correct = tokens_total / max(correct_total, 1)
    avg_latency = mean(latencies)

    summary = {
        "samples": len(results),
        "performance": performance,
        "performance_per_token": performance_per_token,
        "avg_generated_tokens": avg_generated,
        "avg_total_tokens": avg_total,
        "avg_triggers": avg_triggers,
        "tokens_per_correct": tokens_per_correct,
        "total_generated_tokens": tokens_total,
        "avg_latency_sec": avg_latency,
    }

    print("=== Ablation Summary ===")
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "args": vars(args),
            "summary": summary,
            "records": results,
        }
        args.output.write_text(json.dumps(payload, indent=2))
        logger.info("Wrote results to %s", args.output)


if __name__ == "__main__":
    main()
