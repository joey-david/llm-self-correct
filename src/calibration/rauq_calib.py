#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import yaml

ROOT = Path(__file__).resolve().parent
for candidate in (ROOT, ROOT.parent):
    if candidate.exists() and str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from rauq_minimal import (  # noqa: E402
    AnswerScorer,
    HeadSelector,
    ModelAdapter,
    PromptBuilder,
    RAUQ,
    Runner,
)
from rauq_minimal.runner import set_seed  # noqa: E402

DEFAULT_CONFIG: Dict[str, Any] = {
    "in": "data/calibration/calibration_ds.jsonl",
    "out": "data/artifacts/rauq_output.jsonl",
    "model": "Qwen/Qwen3-8B",
    "alpha": 0.2,  # as recommended in the paper
    "max_new": 256,  # to ensure full answers
    "device": "auto",
    "dtype": "auto",
    "attn_implementation": "eager", # to ensure attention weights are available
    "output_attentions": True,
    "use_chat_template": True,
    "trust_remote_code": None,
    "store_all_heads": False,
    "seed": 42,  # the normie seed
    "dataset_fraction": 1.0, # all of it
    "head_calibration_samples": 32,
    "layer_band": "auto", # can be "auto", None, or (low, high) - the band of layers to use for RAUQ
    "debug_decode": False,
    "debug_progress": False,
}
REQUIRED_KEYS = ("in", "out")


def _coerce_optional_bool(value: Any) -> Optional[bool]:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "y", "on"}:
            return True
        if lowered in {"false", "0", "no", "n", "off"}:
            return False
    raise ValueError(f"Cannot interpret {value!r} as optional boolean")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal RAUQ runner")
    parser.add_argument("--config", help="Path to YAML config file")
    args = parser.parse_args()
    if args.config is None:
        # Use default config if no config file is provided
        args.config = None
    return args

def load_config(config_path: Path) -> Dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as fh:
        loaded = yaml.safe_load(fh) or {}

    if not isinstance(loaded, dict):
        raise ValueError("Config file must contain a mapping at the top level")

    config: Dict[str, Any] = DEFAULT_CONFIG.copy()
    config.update(loaded)

    fraction = config.get("dataset_fraction", DEFAULT_CONFIG["dataset_fraction"])
    if fraction is None:
        fraction = DEFAULT_CONFIG["dataset_fraction"]
    try:
        fraction_val = float(fraction)
    except (TypeError, ValueError) as exc:
        raise ValueError("dataset_fraction must be a float between 0 and 1") from exc
    # clamp
    fraction_val = max(0.0, min(1.0, fraction_val))
    config["dataset_fraction"] = fraction_val

    calib_value = config.get(
        "head_calibration_samples", DEFAULT_CONFIG["head_calibration_samples"]
    )
    try:
        calib_int = int(calib_value)
    except (TypeError, ValueError) as exc:
        raise ValueError("head_calibration_samples must be an integer >= 0") from exc
    if calib_int < 0:
        raise ValueError("head_calibration_samples must be an integer >= 0")
    config["head_calibration_samples"] = calib_int

    missing = [key for key in REQUIRED_KEYS if not config.get(key)]
    if missing:
        raise ValueError(f"Config missing required field(s): {', '.join(missing)}")

    return config


def main() -> None:
    args = parse_args()
    if args.config:
        config = load_config(Path(args.config))
    else:
        # Use default config when no config file is provided
        config = DEFAULT_CONFIG.copy()
    set_seed(int(config.get("seed", DEFAULT_CONFIG["seed"])))

    attn_impl = config.get("attn_implementation", DEFAULT_CONFIG["attn_implementation"])
    attn_implementation = str(attn_impl) if attn_impl else None

    output_attn_cfg = config.get("output_attentions", DEFAULT_CONFIG["output_attentions"])
    output_attentions_opt = _coerce_optional_bool(output_attn_cfg)
    if output_attentions_opt is None:
        output_attentions = bool(DEFAULT_CONFIG["output_attentions"])
    else:
        output_attentions = output_attentions_opt

    use_chat_template_cfg = config.get("use_chat_template", DEFAULT_CONFIG["use_chat_template"])
    use_chat_template = _coerce_optional_bool(use_chat_template_cfg)

    trust_remote_cfg = config.get("trust_remote_code", DEFAULT_CONFIG["trust_remote_code"])
    trust_remote_code = _coerce_optional_bool(trust_remote_cfg)

    debug_cfg = config.get("debug_decode", DEFAULT_CONFIG["debug_decode"])
    debug_decode_opt = _coerce_optional_bool(debug_cfg)
    if debug_decode_opt is None:
        debug_decode = bool(DEFAULT_CONFIG["debug_decode"])
    else:
        debug_decode = debug_decode_opt

    debug_progress_cfg = config.get("debug_progress", DEFAULT_CONFIG["debug_progress"])
    debug_progress_opt = _coerce_optional_bool(debug_progress_cfg)
    if debug_progress_opt is None:
        debug_progress = bool(DEFAULT_CONFIG["debug_progress"])
    else:
        debug_progress = debug_progress_opt

    # select a band of layers to use - not too deep, not too shallow
    # can be "auto", None, or (low, high)
    layer_band_cfg = config.get("layer_band", DEFAULT_CONFIG["layer_band"])
    auto_layer_band = False
    explicit_layer_band: Optional[Tuple[int, int]] = None
    if isinstance(layer_band_cfg, str):
        if layer_band_cfg.strip().lower() == "auto":
            auto_layer_band = True
        else:
            raise ValueError("layer_band string value must be 'auto'")
    elif layer_band_cfg is None:
        explicit_layer_band = None
    elif isinstance(layer_band_cfg, (list, tuple)) and len(layer_band_cfg) == 2:
        try:
            low = int(layer_band_cfg[0])
            high = int(layer_band_cfg[1])
        except (TypeError, ValueError) as exc:
            raise ValueError("layer_band entries must be integers") from exc
        explicit_layer_band = (low, high)
    else:
        raise ValueError(
            "layer_band must be 'auto', null, or a two-element sequence of integers"
        )

    # initialize components
    model_adapter = ModelAdapter(
        model_name=str(config.get("model", DEFAULT_CONFIG["model"])),
        device=str(config.get("device", DEFAULT_CONFIG["device"])),
        dtype=config.get("dtype", DEFAULT_CONFIG["dtype"]),
        attn_implementation=attn_implementation,
        output_attentions=output_attentions,
        use_chat_template=use_chat_template,
        trust_remote_code=trust_remote_code,
        debug_decode=debug_decode,
        debug_progress=debug_progress,
    )
    prompt_builder = PromptBuilder()
    scorer = AnswerScorer()
    head_selector = HeadSelector()
    resolved_layer_band = explicit_layer_band
    # layer band auto-resolution
    if auto_layer_band:
        num_layers = int(getattr(model_adapter.config, "num_hidden_layers", 0) or 0)
        if num_layers > 0:
            # Paper selects layers from the middle region of the transformer stack.
            third = max(1, num_layers // 3)
            low = max(0, third - 1)
            high = min(num_layers - 1, num_layers - third)
            if high < low:
                high = low
            resolved_layer_band = (low, high)
    rauq = RAUQ(
        alpha=float(config.get("alpha", DEFAULT_CONFIG["alpha"])),
        layer_band=resolved_layer_band,
    )

    # run the rauq calibration
    runner = Runner(
        model_adapter=model_adapter,
        prompt_builder=prompt_builder,
        scorer=scorer,
        head_selector=head_selector,
        rauq=rauq,
        max_new_tokens=int(config.get("max_new", DEFAULT_CONFIG["max_new"])),
        store_all_heads=bool(config.get("store_all_heads", DEFAULT_CONFIG["store_all_heads"])),
        dataset_fraction=float(config.get("dataset_fraction", DEFAULT_CONFIG["dataset_fraction"])),
        head_calibration_samples=int(
            config.get(
                "head_calibration_samples", DEFAULT_CONFIG["head_calibration_samples"]
            )
        ),
        debug_decode=debug_decode,
    )

    runner.run(str(config["in"]), str(config["out"]))
if __name__ == "__main__":
    main()
