"""Metrics for evaluating controller outputs."""
from __future__ import annotations

from typing import Callable, Dict


MetricFn = Callable[[Dict[str, str], str], bool]


def exact_match(sample: Dict[str, str], prediction: str) -> bool:
    return prediction.strip() == sample["reference"].strip()


def gsm8k(sample: Dict[str, str], prediction: str) -> bool:
    return exact_match(sample, prediction)


def math(sample: Dict[str, str], prediction: str) -> bool:
    return exact_match(sample, prediction)


def trivia(sample: Dict[str, str], prediction: str) -> bool:
    return exact_match(sample, prediction)


METRICS: Dict[str, MetricFn] = {
    "gsm8k": gsm8k,
    "math": math,
    "triviaqa": trivia,
}


__all__ = ["MetricFn", "METRICS", "exact_match"]
