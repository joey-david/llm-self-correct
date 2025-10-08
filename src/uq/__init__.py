from importlib import import_module
from typing import Any, Dict, Tuple

__all__ = [
    "HFGenerator",
    "GenerationOutput",
    "rauq_score",
    "RAUQOutput",
    "BaselineResult",
    "get_baseline",
    "msp_perplexity",
    "attention_score",
    "semantic_entropy",
    "prr",
    "roc_auc",
    "accuracy",
    "enforce_determinism",
    "require_gpu",
    "setup_logging",
]

_LAZY_ATTRS: Dict[str, Tuple[str, str]] = {
    "HFGenerator": (".gen", "HFGenerator"),
    "GenerationOutput": (".gen", "GenerationOutput"),
    "rauq_score": (".rauq", "rauq_score"),
    "RAUQOutput": (".rauq", "RAUQOutput"),
    "BaselineResult": (".baselines", "BaselineResult"),
    "get_baseline": (".baselines", "get_baseline"),
    "msp_perplexity": (".baselines", "msp_perplexity"),
    "attention_score": (".baselines", "attention_score"),
    "semantic_entropy": (".baselines", "semantic_entropy"),
    "prr": (".metrics", "prr"),
    "roc_auc": (".metrics", "roc_auc"),
    "accuracy": (".metrics", "accuracy"),
    "enforce_determinism": (".io", "enforce_determinism"),
    "require_gpu": (".io", "require_gpu"),
    "setup_logging": (".io", "setup_logging"),
}


def __getattr__(name: str) -> Any:
    if name not in _LAZY_ATTRS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _LAZY_ATTRS[name]
    module = import_module(module_name, package=__name__)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(list(__all__))
