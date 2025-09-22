"""Load evaluation datasets for RAUQ/CoT experiments."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional

from datasets import load_dataset


@dataclass
class BenchmarkSample:
    prompt: str
    reference: str
    metadata: Dict[str, str]


def _format_gsm8k(example: Dict) -> BenchmarkSample:
    question = example["question"].strip()
    answer = example["answer"].split("####")[-1].strip()
    prompt = f"{question}\nAnswer:"
    return BenchmarkSample(prompt=prompt, reference=answer, metadata={"id": example.get("id", "")})


def _format_math(example: Dict) -> BenchmarkSample:
    prompt = example["problem"].strip()
    reference = example["solution"].strip()
    return BenchmarkSample(prompt=prompt, reference=reference, metadata={"level": example.get("level", "")})


def _format_humaneval(example: Dict) -> BenchmarkSample:
    prompt = example["prompt"]
    reference = example["canonical_solution"]
    return BenchmarkSample(prompt=prompt, reference=reference, metadata={"task_id": example["task_id"]})


_LOADERS: Dict[str, Callable[[Dict], BenchmarkSample]] = {
    "gsm8k": _format_gsm8k,
    "math": _format_math,
    "humaneval": _format_humaneval,
}

_DATASETS: Dict[str, Dict[str, str]] = {
    "gsm8k": {"path": "gsm8k", "name": "main", "split": "test"},
    "math": {"path": "competition_math", "name": None, "split": "test"},
    "humaneval": {"path": "openai_humaneval", "name": None, "split": "test"},
}


def load_benchmark(name: str, split: Optional[str] = None, limit: Optional[int] = None) -> List[BenchmarkSample]:
    if name not in _LOADERS:
        raise ValueError(f"Unsupported benchmark: {name}")
    dataset_args = _DATASETS[name]
    path = dataset_args["path"]
    ds_kwargs = {"split": split or dataset_args["split"]}
    if dataset_args["name"] is not None:
        ds_kwargs["name"] = dataset_args["name"]
    dataset = load_dataset(path, **ds_kwargs)
    formatter = _LOADERS[name]
    samples = [formatter(example) for example in dataset]
    if limit is not None:
        samples = samples[:limit]
    return samples


__all__ = ["BenchmarkSample", "load_benchmark"]
