"""Load evaluation datasets for RAUQ/CoT experiments."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

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


def _format_triviaqa(example: Dict) -> BenchmarkSample:
    """Format TriviaQA samples into `(prompt, reference, metadata)` tuples."""

    question = example.get("question", "").strip()
    answer = ""
    answer_info = example.get("answer") or {}
    if isinstance(answer_info, dict):
        answer = (answer_info.get("value") or "").strip()
        if not answer:
            aliases = answer_info.get("aliases") or []
            if aliases:
                answer = str(aliases[0]).strip()
    if not answer:
        raise ValueError("TriviaQA sample missing answer value")

    # Prefer a short evidence snippet if available, otherwise fall back to the bare question.
    prompt_prefix = ""
    search_results = example.get("search_results") or []
    if search_results:
        first = search_results[0] or {}
        prompt_prefix = (first.get("search_context") or first.get("snippet") or "").strip()
    if not prompt_prefix:
        evidence = example.get("entity_pages") or []
        if evidence:
            first_page = evidence[0] or {}
            prompt_prefix = (first_page.get("wiki_context") or "").strip()

    if prompt_prefix:
        prompt = f"{prompt_prefix}\n\nQuestion: {question}\nAnswer:"
    else:
        prompt = f"Question: {question}\nAnswer:"

    metadata = {"id": example.get("question_id") or example.get("id") or ""}
    return BenchmarkSample(prompt=prompt, reference=answer, metadata=metadata)


_LOADERS: Dict[str, Callable[[Dict], BenchmarkSample]] = {
    "gsm8k": _format_gsm8k,
    "math": _format_math,
    "humaneval": _format_humaneval,
    "triviaqa": _format_triviaqa,
}

_DATASETS: Dict[str, Dict[str, str]] = {
    "gsm8k": {"path": "gsm8k", "name": "main", "split": "test"},
    "math": {"path": "competition_math", "name": None, "split": "test"},
    "humaneval": {"path": "openai_humaneval", "name": None, "split": "test"},
    "triviaqa": {"path": "trivia_qa", "name": "rc", "split": "validation"},
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
