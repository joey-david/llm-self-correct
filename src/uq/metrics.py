from __future__ import annotations

import math
from functools import lru_cache
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np
from sklearn.metrics import roc_auc_score


def prr(scores: Sequence[float], qualities: Sequence[float], top_fraction: float = 0.5) -> float:
    scores_arr = np.asarray(scores, dtype=float)
    qual_arr = np.asarray(qualities, dtype=float)
    n = len(scores_arr)
    if n == 0:
        raise ValueError("PRR requires non-empty inputs")
    k = max(1, int(math.ceil(n * top_fraction)))
    order = np.argsort(-scores_arr)
    total = qual_arr.sum()
    cumsum = np.cumsum(qual_arr[order])
    retained = []
    for j in range(k):
        removed = cumsum[j]
        remain = total - removed
        remain_count = n - (j + 1)
        retained.append(remain / remain_count if remain_count > 0 else qual_arr.mean())
    area_model = float(np.mean(retained))
    baseline = float(qual_arr.mean())
    worst_order = np.argsort(qual_arr)
    worst_cumsum = np.cumsum(qual_arr[worst_order])
    oracle_retained = []
    for j in range(k):
        removed = worst_cumsum[j]
        remain = total - removed
        remain_count = n - (j + 1)
        oracle_retained.append(remain / remain_count if remain_count > 0 else baseline)
    area_oracle = float(np.mean(oracle_retained))
    denom = max(area_oracle - baseline, 1e-9)
    return (area_model - baseline) / denom


def roc_auc(scores: Sequence[float], labels: Sequence[int]) -> float:
    if len(set(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, scores))


def accuracy(preds: Sequence[str], gold: Sequence[str]) -> float:
    if not preds:
        return 0.0
    return sum(int(p.strip() == g.strip()) for p, g in zip(preds, gold)) / len(preds)


@lru_cache(maxsize=1)
def load_alignscore(model_name: str = "yzha/AlignScore"):
    try:
        from alignscore import AlignScore  # type: ignore[attr-defined]
    except ImportError:
        from alignscore.alignscore import AlignScore  # type: ignore[attr-defined]
    model = AlignScore(model=model_name)
    return model


def alignscore_scores(model, source: Sequence[str], generated: Sequence[str], reference: Sequence[str]) -> List[float]:
    scores = model.score(src_list=list(source), hyp_list=list(generated), ref_list=list(reference))
    return list(map(float, scores))


def comet_scores(model, source: Sequence[str], generated: Sequence[str], reference: Sequence[str]) -> List[float]:
    outputs = model.predict(source=source, translation=generated, reference=reference, batch_size=8)
    return [float(o["score"]) for o in outputs]


def plot_scatter(x: Sequence[float], y: Sequence[float], path: Path, xlabel: str, ylabel: str) -> None:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(4, 4))
    plt.scatter(x, y, s=12, alpha=0.7)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
