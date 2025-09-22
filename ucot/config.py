"""Configuration dataclasses for RAUQ-triggered CoT experiments.

The defaults mirror the guidance in technicals.md.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class HeadSelectionConfig:
    """Configuration for selecting uncertainty-aware attention heads."""

    calibration_paths: List[Path]
    model_name: str
    tokenizer_name: Optional[str] = None
    output_path: Path = Path("artifacts/head_selection.json")
    num_examples: Optional[int] = 256
    layers_fraction: float = 0.33
    device: str = "cuda"


@dataclass
class RAUQConfig:
    """Runtime configuration for RAUQ scoring."""

    alpha: float = 0.3
    layers_fraction: float = 0.33
    head_indices_path: Path = Path("artifacts/head_selection.json")
    device: str = "cuda"
    eps: float = 1e-12


@dataclass
class ThresholdTrainingConfig:
    """Configuration for learning the RAUQ trigger threshold."""

    calibration_paths: List[Path]
    model_name: str
    tokenizer_name: Optional[str] = None
    head_indices_path: Path = Path("artifacts/head_selection.json")
    output_path: Path = Path("artifacts/threshold.json")
    alpha: float = 0.3
    max_samples: Optional[int] = 2048
    batch_size: int = 4
    device: str = "cuda"
    logistic_penalty: float = 1.0


@dataclass
class RollbackConfig:
    """Parameters controlling rollback behaviour."""

    stability_window: int = 2 # num. of tokens to consider for stability
    rollback_depth: int = 3 # num. of tokens to erase on rollback
    cooldown: int = 5 # num. of tokens to wait after a rollback
    max_triggers: Optional[int] = 5 # max num. of rollbacks per generation
    mode: str = "fixed" # "fixed" uses rollback_depth, "anchor" jumps to stability anchor


@dataclass
class CoTConfig:
    """Parameters for micro-CoT resampling."""

    max_cot_tokens: int = 200
    cot_prefix: str = ""
    candidates: int = 3
    lookahead_horizon: int = 4
    length_penalty: float = 2e-3
    temperature: float = 0.7
    top_p: float = 0.95
    stop_mode: str = "fixed" # "fixed" respects max_cot_tokens, "rauq" stops on stability, "none" disables early stop
    # Automatically set the CoT prefix based on max_cot_tokens
    def __post_init__(self):
        self.cot_prefix = f"Wait, let's quickly think step by step about this (<{self.max_cot_tokens} tokens)."


@dataclass
class RerankConfig:
    """Parameters for pause-and-rerank repair."""

    candidates: int = 3
    lookahead_horizon: int = 4
    temperature: float = 0.7
    top_p: float = 0.95


@dataclass
class ControllerConfig:
    """Full configuration for the RAUQ-guided decoding controller."""

    model_name: str
    tokenizer_name: Optional[str] = None
    max_length: int = 2048
    temperature: float = 0.0
    top_p: float = 1.0
    alpha: float = 0.3
    theta: Optional[float] = None
    head_indices_path: Path = Path("artifacts/head_selection.json")
    threshold_path: Optional[Path] = Path("artifacts/threshold.json")
    rollback: RollbackConfig = field(default_factory=RollbackConfig)
    cot: CoTConfig = field(default_factory=CoTConfig)
    rerank: RerankConfig = field(default_factory=RerankConfig)
    repair_strategy: str = "cot" # "cot" uses micro-CoT, "rerank" uses pause+rerank
    device: str = "cuda"
    max_new_tokens: int = 1024
    stop_sequences: Optional[List[str]] = None


@dataclass
class AblationConfig:
    """Configuration wrapper for ablations."""

    trigger: str = "rauq"
    repair: str = "cot"
    rollback: int = 2
    theta: Optional[float] = None
    alpha: float = 0.3
    cot_stop: str = "rauq"
    dataset: str = "gsm8k"
    output_dir: Path = Path("artifacts/ablations")


@dataclass
class FinetuneConfig:
    """Configuration for QLoRA finetuning (v2)."""

    model_name: str
    tokenizer_name: Optional[str] = None
    train_path: Path = Path("data/finetune/train.jsonl")
    eval_path: Optional[Path] = None
    output_dir: Path = Path("artifacts/qlora")
    alpha: float = 0.3
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: Optional[List[str]] = None
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.03
    weight_decay: float = 0.0
    num_epochs: float = 2.0
    max_steps: Optional[int] = None
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    bf16: bool = True
    logging_steps: int = 10
    save_steps: int = 200
    max_grad_norm: float = 1.0
    gradient_checkpointing: bool = True
    seed: int = 1037


__all__ = [
    "HeadSelectionConfig",
    "RAUQConfig",
    "ThresholdTrainingConfig",
    "RollbackConfig",
    "CoTConfig",
    "RerankConfig",
    "ControllerConfig",
    "AblationConfig",
    "FinetuneConfig",
]
