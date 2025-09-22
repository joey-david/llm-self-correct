"""Synthetic calibration corpora for RAUQ head/threshold fitting.

These helpers provide lightweight prompt/completion pairs that are
disjoint from our benchmark splits, so calibrating RAUQ will not leak
evaluation data.  The default generator focuses on small arithmetic
word problems that exercise short-chain reasoning without being
benchmark questions.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, List, Sequence, Tuple

import random


Sample = Tuple[str, str]


@dataclass(frozen=True)
class SyntheticCorpusConfig:
    """Configuration for generating a synthetic calibration corpus."""

    num_samples: int = 512
    seed: int = 1037


def _generate_addition(rng: random.Random) -> Sample:
    a = rng.randint(12, 999)
    b = rng.randint(12, 999)
    prompt = f"Solve the addition problem: {a} + {b} ="
    completion = str(a + b)
    return prompt, completion


def _generate_subtraction(rng: random.Random) -> Sample:
    a = rng.randint(200, 999)
    b = rng.randint(12, a - 5)
    prompt = f"Compute the difference: {a} - {b} ="
    completion = str(a - b)
    return prompt, completion


def _generate_multiplication(rng: random.Random) -> Sample:
    a = rng.randint(6, 45)
    b = rng.randint(6, 45)
    prompt = f"What is {a} × {b}?"
    completion = str(a * b)
    return prompt, completion


def _generate_division(rng: random.Random) -> Sample:
    divisor = rng.randint(3, 25)
    quotient = rng.randint(3, 40)
    dividend = divisor * quotient
    prompt = f"Divide {dividend} by {divisor}."
    completion = str(quotient)
    return prompt, completion


def _generate_mixed(rng: random.Random) -> Sample:
    a = rng.randint(10, 99)
    b = rng.randint(2, 20)
    c = rng.randint(2, 15)
    prompt = f"Evaluate ({a} + {b}) × {c}."
    completion = str((a + b) * c)
    return prompt, completion


_GENERATORS = (
    _generate_addition,
    _generate_subtraction,
    _generate_multiplication,
    _generate_division,
    _generate_mixed,
)


def generate_synthetic_corpus(config: SyntheticCorpusConfig | None = None) -> List[Sample]:
    """Create a list of prompt/completion pairs for calibration."""

    cfg = config or SyntheticCorpusConfig()
    rng = random.Random(cfg.seed)
    samples: List[Sample] = []
    for _ in range(cfg.num_samples):
        generator = rng.choice(_GENERATORS)
        samples.append(generator(rng))
    return samples


def write_corpus(samples: Sequence[Sample], path: Path) -> None:
    """Persist the synthetic corpus as newline-delimited JSON."""

    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fp:
        for prompt, completion in samples:
            payload = {"prompt": prompt, "completion": completion}
            fp.write(json.dumps(payload) + "\n")


__all__ = ["SyntheticCorpusConfig", "generate_synthetic_corpus", "write_corpus"]
