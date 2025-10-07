from __future__ import annotations

import json
import logging
import os
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import yaml

GLOBAL_SEED = 20251006


def enforce_determinism(seed: int = GLOBAL_SEED) -> None:
    random.seed(seed)
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    try:  # numpy is optional
        import numpy as np  # type: ignore

        np.random.seed(seed)
    except ModuleNotFoundError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)
    except ModuleNotFoundError:
        pass


def require_gpu(force_cpu: bool = False) -> None:
    if force_cpu:
        return
    try:
        import torch
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise RuntimeError("torch must be installed before running the pipeline") from exc
    if not torch.cuda.is_available():
        raise SystemExit("GPU_NOT_FOUND: aborting to respect no-CPU-inference policy.")
    device = torch.cuda.get_device_properties(0)
    logging.getLogger(__name__).info(
        "Using GPU %s (%0.2f GiB)", device.name, device.total_memory / 2**30
    )


def setup_logging(log_dir: Path) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "run.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)sZ %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler(),
        ],
    )


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in records:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def timestamp() -> str:
    return datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
