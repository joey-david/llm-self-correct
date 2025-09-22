#!/usr/bin/env python
"""Run the RAUQ-triggered controller on an input prompt."""
from __future__ import annotations

import argparse
from pathlib import Path

from ucot.config import ControllerConfig, RAUQConfig
from ucot.controller import RAUQController
from ucot.rauq import RAUQScorer
from ucot.uncertainty import RAUQScorerWrapper
from ucot.threshold import ThresholdResult
from ucot.utils.logging import setup_logging
from ucot.utils.model import load_model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run RAUQ-triggered rollback + CoT controller")
    parser.add_argument("--prompt", required=True, help="Prompt text or path to file")
    parser.add_argument("--model", required=True, help="Model name or path")
    parser.add_argument("--tokenizer", help="Tokenizer name (defaults to model)")
    parser.add_argument("--heads", type=Path, default=Path("artifacts/head_selection.json"))
    parser.add_argument("--threshold", type=Path, default=Path("artifacts/threshold.json"))
    parser.add_argument("--theta", type=float, help="Override threshold value")
    parser.add_argument("--alpha", type=float, default=0.3)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cot-temperature", type=float, default=0.7)
    parser.add_argument("--cot-top-p", type=float, default=0.95)
    parser.add_argument("--cot-length", type=int, default=20)
    parser.add_argument("--cot-horizon", type=int, default=4)
    parser.add_argument("--cot-candidates", type=int, default=3)
    parser.add_argument("--cot-mode", choices=["fixed", "rauq", "none"], default="fixed")
    parser.add_argument("--repair", choices=["cot", "rerank", "none"], default="cot")
    parser.add_argument("--rerank-candidates", type=int, default=3)
    parser.add_argument("--rerank-temperature", type=float, default=0.7)
    parser.add_argument("--rerank-top-p", type=float, default=0.95)
    parser.add_argument("--rerank-horizon", type=int, default=4)
    parser.add_argument("--rollback", type=int, default=2)
    parser.add_argument("--rollback-mode", choices=["fixed", "anchor"], default="fixed")
    parser.add_argument("--cooldown", type=int, default=5)
    parser.add_argument("--stability-window", type=int, default=2)
    parser.add_argument("--max-triggers", type=int, default=5)
    return parser


def load_prompt(raw: str) -> str:
    path = Path(raw)
    if path.exists():
        return path.read_text()
    return raw


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    logger = setup_logging()

    prompt = load_prompt(args.prompt)
    loaded = load_model(
        model_name=args.model,
        tokenizer_name=args.tokenizer,
        device=args.device,
    )

    rauq_config = RAUQConfig(alpha=args.alpha, head_indices_path=args.heads, device=args.device)
    base_scorer = RAUQScorer.from_config(rauq_config, num_layers=loaded.model.config.num_hidden_layers)
    scorer = RAUQScorerWrapper(
        alpha=base_scorer.alpha,
        head_indices=base_scorer.head_indices,
        layers=base_scorer.layers,
        eps=base_scorer.eps,
        device=base_scorer.device,
    )
    threshold = None
    if args.threshold.exists():
        threshold = ThresholdResult.load(args.threshold)
        logger.info("Loaded threshold θ=%.4f from %s", threshold.theta, args.threshold)
    if args.theta is not None:
        if threshold:
            threshold.theta = args.theta
        else:
            threshold = ThresholdResult(theta=args.theta, logistic_coef=0.0, logistic_intercept=0.0)

    controller_config = ControllerConfig(
        model_name=args.model,
        tokenizer_name=args.tokenizer,
        theta=args.theta if threshold is None else threshold.theta,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    controller_config.cot.max_cot_tokens = args.cot_length
    controller_config.cot.lookahead_horizon = args.cot_horizon
    controller_config.cot.temperature = args.cot_temperature
    controller_config.cot.top_p = args.cot_top_p
    controller_config.cot.candidates = args.cot_candidates
    controller_config.cot.stop_mode = args.cot_mode
    controller_config.cot.cot_prefix = (
        f"Wait, let's quickly think step by step about this (<{controller_config.cot.max_cot_tokens} tokens)."
    )
    controller_config.repair_strategy = args.repair
    controller_config.rerank.candidates = args.rerank_candidates
    controller_config.rerank.lookahead_horizon = args.rerank_horizon
    controller_config.rerank.temperature = args.rerank_temperature
    controller_config.rerank.top_p = args.rerank_top_p
    controller_config.rollback.rollback_depth = args.rollback
    controller_config.rollback.mode = args.rollback_mode
    controller_config.rollback.cooldown = args.cooldown
    controller_config.rollback.stability_window = args.stability_window
    controller_config.rollback.max_triggers = args.max_triggers

    controller = RAUQController(
        loaded=loaded,
        config=controller_config,
        scorer=scorer,
        threshold=threshold,
    )
    output = controller.generate(prompt)

    logger.info("Completion: %s", output.completion)
    logger.info("Triggers: %d", len(output.trigger_events))
    logger.info("RAUQ trace (first 10): %s", output.rauq_scores[:10])


if __name__ == "__main__":
    main()
