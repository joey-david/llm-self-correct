from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Union

import torch
from tqdm.auto import tqdm

from ..datasets import load_dataset
from ..aspects import SpikeConfig, detect_spikes
from .baselines import attention_score, msp_perplexity
from .gen import HFGenerator
from .io import enforce_determinism, load_yaml, require_gpu, setup_logging, timestamp, write_jsonl
from .metrics import accuracy, prr, roc_auc
from .rauq import RAUQOutput, rauq_score


@dataclass
class EvalResult:
    metrics: Dict[str, float]
    output_dir: Path


def _default_out(cfg: dict) -> Path:
    root = Path(cfg.get("output_root", "data/experiments"))
    return root / timestamp()


def run_eval(config_path: Path | str, output_dir: Path | None = None) -> EvalResult:
    cfg = load_yaml(Path(config_path))
    enforce_determinism()
    force_cpu = bool(cfg.get("force_cpu", False))
    require_gpu(force_cpu=force_cpu)
    out_dir = Path(output_dir or cfg.get("output_dir", _default_out(cfg)))
    setup_logging(out_dir)
    model_cfg = cfg.get("model", {})
    torch_dtype_cfg: Union[str, torch.dtype, None] = model_cfg.get("torch_dtype")
    if isinstance(torch_dtype_cfg, str):
        dtype_attr = torch_dtype_cfg.strip().lower()
        if dtype_attr == "auto":
            torch_dtype = torch_dtype_cfg  # type: ignore[assignment]
        else:
            try:
                torch_dtype = getattr(torch, dtype_attr)
            except AttributeError as exc:
                raise ValueError(f"Unsupported torch dtype requested: {torch_dtype_cfg}") from exc
    elif torch_dtype_cfg is None or isinstance(torch_dtype_cfg, torch.dtype):
        torch_dtype = torch_dtype_cfg
    else:
        raise TypeError(f"torch_dtype must be a string or torch.dtype, got {type(torch_dtype_cfg).__name__}")
    generator = HFGenerator(
        model_cfg.get("name", "meta-llama/Meta-Llama-3.1-8B"),
        force_cpu=force_cpu,
        torch_dtype=torch_dtype if torch_dtype != "auto" else None,
    )
    max_tokens = int(model_cfg.get("max_new_tokens", 256))
    rauq_cfg = cfg.get("rauq", {})
    alpha = float(rauq_cfg.get("alpha", 0.2))
    layers = rauq_cfg.get("layers")
    spike_cfg = SpikeConfig(**cfg.get("gating", {}).get("spike", {}))
    dataset_cfgs = cfg.get("datasets", [])
    predictions_path = out_dir / "predictions.jsonl"
    rauq_path = out_dir / "scores_rauq.jsonl"
    baselines_path = out_dir / "scores_baselines.jsonl"
    metrics: Dict[str, List[float]] = {"rauq": [], "msp": [], "attention": []}
    qualities: List[float] = []
    labels: List[int] = []
    preds: List[str] = []
    refs: List[str] = []
    scores_records = []
    baseline_records = []
    pred_records = []
    dataset_iter = tqdm(dataset_cfgs, desc="Datasets", unit="dataset")
    for ds_cfg in dataset_iter:
        name = ds_cfg["name"]
        dataset_iter.set_postfix_str(name)
        logging.info("Running dataset %s", name)
        opts = {k: v for k, v in ds_cfg.items() if k != "name"}
        config_path = opts.pop("config", None)
        if config_path:
            base = load_yaml(Path(config_path))
            base.update(opts)
            opts = base
        examples = load_dataset(name, opts)
        example_iter = tqdm(examples, desc=name, unit="sample", leave=False)
        for ex in example_iter:
            out = generator.generate(ex.prompt, max_new_tokens=max_tokens)
            rauq_out = rauq_score(out.token_probs, out.attentions, out.prompt_len, alpha=alpha, layers_subset=layers)
            bas_msp = msp_perplexity(out.token_probs)
            bas_att = attention_score(out.attentions, out.prompt_len)
            spikes = detect_spikes(rauq_out.token_spikes, spike_cfg)
            record = {
                "id": ex.uid,
                "dataset": name,
                "prompt": ex.prompt,
                "gold": ex.reference,
                "prediction": out.text,
                "token_probs": out.token_probs.tolist(),
                "selected_heads": rauq_out.selected_heads,
                "spikes": spikes,
            }
            pred_records.append(record)
            scores_records.append({"id": ex.uid, "dataset": name, "u": rauq_out.u, "token_spikes": rauq_out.token_spikes})
            baseline_records.append(
                {
                    "id": ex.uid,
                    "dataset": name,
                    "msp": bas_msp.sequence,
                    "attention": bas_att.sequence,
                }
            )
            metrics["rauq"].append(rauq_out.u)
            metrics["msp"].append(bas_msp.sequence)
            metrics["attention"].append(bas_att.sequence)
            preds.append(out.text)
            refs.append(ex.reference)
            qualities.append(1.0 if out.text.strip() == ex.reference.strip() else 0.0)
            labels.append(int(out.text.strip() != ex.reference.strip()))
    write_jsonl(predictions_path, pred_records)
    write_jsonl(rauq_path, scores_records)
    write_jsonl(baselines_path, baseline_records)
    overall_metrics = {
        "prr": prr(metrics["rauq"], qualities),
        "roc_auc": roc_auc(metrics["rauq"], labels),
        "accuracy": accuracy(preds, refs),
    }
    with (out_dir / "metrics.json").open("w", encoding="utf-8") as fh:
        json.dump(overall_metrics, fh, indent=2)
    return EvalResult(metrics=overall_metrics, output_dir=out_dir)
