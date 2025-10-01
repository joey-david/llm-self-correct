from __future__ import annotations

import argparse
import json
import math
import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

import matplotlib

# Use a non-interactive backend for headless environments
matplotlib.use("Agg")
import matplotlib.pyplot as plt



@dataclass
class Record:
    id: str
    dataset: str
    prompt: str
    pred_text: str
    answers: List[str]
    correct: Optional[bool]
    alignscore_best: Optional[float]
    alpha: Optional[float]
    selected_layer: Optional[str]  # like "l29"
    selected_head: Optional[int]
    u_token: List[float]
    u_final: Optional[float]
    # Optional extras (if present when store_all_heads=True)
    selected_heads: Optional[Dict[str, int]] = None


def load_jsonl(path: Path) -> List[Record]:
    out: List[Record] = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except json.JSONDecodeError:
                continue
            try:
                rec = Record(
                    id=str(obj.get("id", f"<missing:{i}>")),
                    dataset=str(obj.get("dataset", "")),
                    prompt=str(obj.get("prompt", "")),
                    pred_text=str(obj.get("pred_text", "")),
                    answers=[a for a in (obj.get("answers") or []) if isinstance(a, str)],
                    correct=(
                        bool(obj.get("correct"))
                        if obj.get("correct") is not None
                        else None
                    ),
                    alignscore_best=(
                        float(obj["alignscore_best"]) if obj.get("alignscore_best") is not None else None
                    ),
                    alpha=(float(obj["alpha"]) if obj.get("alpha") is not None else None),
                    selected_layer=(obj.get("selected_layer") if obj.get("selected_layer") is not None else None),
                    selected_head=(
                        int(obj["selected_head"]) if obj.get("selected_head") is not None else None
                    ),
                    u_token=[float(x) for x in (obj.get("u_token") or []) if _is_finite(x)],
                    u_final=(float(obj["u_final"]) if obj.get("u_final") is not None else None),
                    selected_heads=(
                        obj.get("selected_heads") if isinstance(obj.get("selected_heads"), dict) else None
                    ),
                )
                out.append(rec)
            except Exception:
                # Skip malformed records
                continue
    return out


def _is_finite(x: Any) -> bool:
    try:
        return math.isfinite(float(x))
    except Exception:
        return False


def _max_finite(seq: Sequence[float]) -> Optional[float]:
    best: Optional[float] = None
    for v in seq:
        try:
            vf = float(v)
        except Exception:
            continue
        if not math.isfinite(vf):
            continue
        if best is None or vf > best:
            best = vf
    return best


def _collect_scores(
    records: Iterable[Record],
    value_fn,
) -> Tuple[np.ndarray, np.ndarray]:
    xs: List[float] = []
    ys: List[int] = []
    for r in records:
        if r.correct is None:
            continue
        val = value_fn(r)
        if val is None:
            continue
        try:
            fv = float(val)
        except Exception:
            continue
        if not math.isfinite(fv):
            continue
        xs.append(fv)
        ys.append(1 if not r.correct else 0)
    return np.array(xs, dtype=float), np.array(ys, dtype=int)


# ------------------------------- Metrics utils -----------------------------


def accuracy(labels: Sequence[bool]) -> float:
    if not labels:
        return float("nan")
    return float(np.mean(np.array(labels, dtype=float)))


def roc_pr_from_scores(y_true_incorrect: np.ndarray, scores: np.ndarray) -> Tuple[
    Tuple[np.ndarray, np.ndarray, float],  # (fpr, tpr, auc)
    Tuple[np.ndarray, np.ndarray, float],  # (recall, precision, ap)
]:
    """
    Compute ROC and PR data for detecting incorrect examples.
    y_true_incorrect: 1 for incorrect (positive), 0 for correct (negative)
    scores: higher => more likely incorrect (e.g., u_final)
    Returns: (fpr, tpr, auc), (recall, precision, ap)
    """
    assert y_true_incorrect.shape == scores.shape
    n_pos = int(np.sum(y_true_incorrect == 1))
    n_neg = int(np.sum(y_true_incorrect == 0))
    if n_pos == 0 or n_neg == 0:
        # Degenerate: cannot compute ROC/PR
        return (
            np.array([0.0, 1.0]),
            np.array([0.0, 1.0]),
            float("nan"),
        ), (
            np.array([0.0, 1.0]),
            np.array([float(n_pos) / max(1, (n_pos + n_neg))] * 2),
            float("nan"),
        )

    # Sort by descending score (more positive first)
    order = np.argsort(-scores, kind="mergesort")
    y = y_true_incorrect[order]
    s = scores[order]

    # Cumulative TPs/FPs at each threshold step
    tps = np.cumsum(y == 1)
    fps = np.cumsum(y == 0)
    thresh_changes = np.where(np.diff(s, prepend=np.inf) != 0)[0]
    tps = tps[thresh_changes]
    fps = fps[thresh_changes]

    fns = n_pos - tps
    tns = n_neg - fps

    tpr = tps / max(1, n_pos)
    fpr = fps / max(1, n_neg)

    # Add (0,0) and (1,1)
    fpr = np.concatenate([[0.0], fpr, [1.0]])
    tpr = np.concatenate([[0.0], tpr, [1.0]])
    # AUC via trapezoidal rule
    auc = float(np.trapz(tpr, fpr))

    # PR curve: precision = TP/(TP+FP), recall = TP/P
    precision = tps / np.maximum(1, (tps + fps))
    recall = tps / max(1, n_pos)
    # Add endpoints (recall=0->precision=pos_rate, recall=1)
    pos_rate = n_pos / (n_pos + n_neg)
    recall = np.concatenate([[0.0], recall, [1.0]])
    precision = np.concatenate([[pos_rate], precision, [precision[-1] if precision.size else pos_rate]])
    # AP via trapezoidal area under P(R) with recall increasing
    ap = float(np.trapz(precision, recall))

    return (fpr, tpr, auc), (recall, precision, ap)


def binned_curve(x: np.ndarray, y: np.ndarray, bins: int = 10) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute binned x centers and mean of y per bin. Returns (centers, means, counts)
    Ignores NaNs.
    """
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size == 0:
        return np.array([]), np.array([]), np.array([])
    edges = np.quantile(x, np.linspace(0.0, 1.0, bins + 1))
    # Make edges strictly increasing to avoid zero-width bins
    for i in range(1, len(edges)):
        if edges[i] <= edges[i - 1]:
            edges[i] = edges[i - 1] + 1e-12
    inds = np.digitize(x, edges[1:-1], right=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    means = np.zeros(bins, dtype=float)
    counts = np.zeros(bins, dtype=int)
    for b in range(bins):
        sel = inds == b
        if np.any(sel):
            means[b] = float(np.mean(y[sel]))
            counts[b] = int(np.sum(sel))
        else:
            means[b] = np.nan
            counts[b] = 0
    return centers, means, counts


def pearsonr_safe(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if not np.any(mask):
        return float("nan")
    xv, yv = x[mask], y[mask]
    if xv.size < 2:
        return float("nan")
    x0 = xv - np.mean(xv)
    y0 = yv - np.mean(yv)
    denom = float(np.sqrt(np.sum(x0 * x0) * np.sum(y0 * y0)))
    if denom == 0:
        return float("nan")
    return float(np.sum(x0 * y0) / denom)


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


# --------------------------------- Plotting --------------------------------


def plot_acc_by_dataset(records: List[Record], outdir: Path) -> Optional[Path]:
    per_ds: Dict[str, List[bool]] = defaultdict(list)
    for r in records:
        if r.correct is None:
            continue
        per_ds[r.dataset or "<unknown>"] .append(bool(r.correct))
    if not per_ds:
        return None
    ds_names = sorted(per_ds.keys())
    accs = [accuracy(per_ds[name]) for name in ds_names]
    counts = [len(per_ds[name]) for name in ds_names]

    plt.figure(figsize=(10, 5), dpi=150)
    bars = plt.bar(range(len(ds_names)), accs, color="#1f77b4", alpha=0.8)
    plt.xticks(range(len(ds_names)), ds_names, rotation=45, ha="right")
    plt.ylim(0, 1)
    plt.ylabel("Accuracy")
    plt.title("Accuracy by dataset")
    # annotate counts
    for rect, n in zip(bars, counts):
        h = rect.get_height()
        plt.text(rect.get_x() + rect.get_width() / 2, h + 0.01, f"n={n}", ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    outpath = outdir / "acc_by_dataset.png"
    plt.savefig(outpath)
    plt.close()
    return outpath


def plot_u_hist_by_correct(records: List[Record], outdir: Path) -> Optional[Path]:
    xs_ok = [r.u_final for r in records if r.u_final is not None and r.correct is True]
    xs_bad = [r.u_final for r in records if r.u_final is not None and r.correct is False]
    xs_ok = [float(x) for x in xs_ok if _is_finite(x)]
    xs_bad = [float(x) for x in xs_bad if _is_finite(x)]
    if not xs_ok and not xs_bad:
        return None
    plt.figure(figsize=(7, 5), dpi=150)
    bins = 40
    if xs_ok:
        plt.hist(xs_ok, bins=bins, alpha=0.6, density=True, label="correct", color="#2ca02c")
    if xs_bad:
        plt.hist(xs_bad, bins=bins, alpha=0.6, density=True, label="incorrect", color="#d62728")
    plt.xlabel("u_final (higher => more uncertain)")
    plt.ylabel("Density")
    plt.title("u_final distribution by correctness")
    plt.legend()
    plt.tight_layout()
    outpath = outdir / "u_final_hist_by_correct.png"
    plt.savefig(outpath)
    plt.close()
    return outpath


def plot_max_u_hist_by_correct(records: List[Record], outdir: Path) -> Optional[Path]:
    vals_ok: List[float] = []
    vals_bad: List[float] = []
    for r in records:
        if not r.u_token:
            continue
        max_u = _max_finite(r.u_token)
        if max_u is None:
            continue
        if r.correct is False:
            vals_bad.append(max_u)
        elif r.correct is True:
            vals_ok.append(max_u)
    if not vals_ok and not vals_bad:
        return None
    plt.figure(figsize=(7, 5), dpi=150)
    bins = 40
    hist_range = (0.0, 3.5)
    if vals_ok:
        plt.hist(
            vals_ok,
            bins=bins,
            range=hist_range,
            alpha=0.6,
            density=True,
            label="correct",
            color="#2ca02c",
        )
    if vals_bad:
        plt.hist(
            vals_bad,
            bins=bins,
            range=hist_range,
            alpha=0.6,
            density=True,
            label="incorrect",
            color="#d62728",
        )
    plt.xlim(hist_range)
    plt.xlabel("max u_token")
    plt.ylabel("Density")
    plt.title("Max token uncertainty by correctness")
    plt.legend()
    plt.tight_layout()
    outpath = outdir / "max_u_token_hist_by_correct.png"
    plt.savefig(outpath)
    plt.close()
    return outpath


def plot_u_final_vs_max(records: List[Record], outdir: Path) -> Optional[Path]:
    xs: List[float] = []
    ys: List[float] = []
    colors: List[str] = []
    for r in records:
        if r.u_final is None or not r.u_token:
            continue
        max_u = _max_finite(r.u_token)
        if max_u is None:
            continue
        uf = float(r.u_final)
        if not math.isfinite(uf):
            continue
        xs.append(max_u)
        ys.append(uf)
        colors.append("#d62728" if r.correct is False else "#2ca02c")
    if not xs:
        return None
    xv = np.array(xs, dtype=float)
    yv = np.array(ys, dtype=float)
    corr = pearsonr_safe(xv, yv)

    plt.figure(figsize=(7, 5), dpi=150)
    plt.scatter(xv, yv, s=16, c=colors, alpha=0.6, edgecolors="none")
    lim = max(float(np.nanmax(xv)), float(np.nanmax(yv))) if xv.size and yv.size else None
    if lim is not None and math.isfinite(lim):
        plt.plot([0, lim], [0, lim], linestyle="--", color="#888888", label="y=x")
    plt.xlabel("max u_token")
    plt.ylabel("u_final")
    plt.title(f"u_final vs max token uncertainty (Pearson r={corr:.3f})")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    outpath = outdir / "u_final_vs_max_token.png"
    plt.savefig(outpath)
    plt.close()
    return outpath


def plot_roc_pr(records: List[Record], outdir: Path) -> Tuple[Optional[Path], Optional[Path]]:
    # RAUQ (u_final)
    u_scores, y = _collect_scores(records, lambda r: r.u_final)
    have_u = u_scores.size >= 2 and np.unique(y).size >= 2
    if have_u:
        (fpr_u, tpr_u, auc_u), (rec_u, prec_u, ap_u) = roc_pr_from_scores(y, u_scores)
    else:
        fpr_u = tpr_u = rec_u = prec_u = np.array([])
        auc_u = ap_u = float("nan")

    # Max token uncertainty
    max_scores, y_max = _collect_scores(records, lambda r: _max_finite(r.u_token or []))
    have_max = max_scores.size >= 2 and np.unique(y_max).size >= 2
    if have_max:
        (fpr_max, tpr_max, auc_max), (rec_max, prec_max, ap_max) = roc_pr_from_scores(y_max, max_scores)
    else:
        fpr_max = tpr_max = rec_max = prec_max = np.array([])
        auc_max = ap_max = float("nan")

    roc_path = pr_path = None
    if fpr_u.size or fpr_max.size:
        plt.figure(figsize=(6, 6), dpi=150)
        if fpr_u.size:
            plt.plot(fpr_u, tpr_u, label=f"u_final (AUC={auc_u:.3f})", color="#1f77b4")
        if fpr_max.size:
            plt.plot(fpr_max, tpr_max, label=f"max u_token (AUC={auc_max:.3f})", color="#ff7f0e")
        plt.plot([0, 1], [0, 1], linestyle="--", color="#888888", label="random")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC: detect incorrect answers")
        plt.legend()
        plt.tight_layout()
        roc_path = outdir / "roc_incorrect.png"
        plt.savefig(roc_path)
        plt.close()

    if rec_u.size or rec_max.size:
        plt.figure(figsize=(6, 6), dpi=150)
        if rec_u.size:
            plt.plot(rec_u, prec_u, label=f"u_final (AP={ap_u:.3f})", color="#1f77b4")
        if rec_max.size:
            plt.plot(rec_max, prec_max, label=f"max u_token (AP={ap_max:.3f})", color="#ff7f0e")
        plt.xlabel("Recall (incorrect)")
        plt.ylabel("Precision (incorrect)")
        plt.title("PR: detect incorrect answers")
        plt.legend()
        plt.tight_layout()
        pr_path = outdir / "pr_incorrect.png"
        plt.savefig(pr_path)
        plt.close()

    return roc_path, pr_path


def plot_binned_failure_rate(records: List[Record], outdir: Path) -> Optional[Path]:
    xs = np.array([float(r.u_final) for r in records if r.u_final is not None and r.correct is not None], dtype=float)
    ys = np.array([1.0 if not r.correct else 0.0 for r in records if r.u_final is not None and r.correct is not None], dtype=float)
    if xs.size == 0:
        return None
    centers, means, counts = binned_curve(xs, ys, bins=12)
    if centers.size == 0:
        return None
    plt.figure(figsize=(7, 5), dpi=150)
    plt.plot(centers, means, marker="o", color="#d62728", label="Empirical failure rate")
    plt.xlabel("u_final (binned)")
    plt.ylabel("Fraction incorrect")
    plt.title("Observed failure rate vs u_final bins")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    outpath = outdir / "failure_rate_vs_u_binned.png"
    plt.savefig(outpath)
    plt.close()
    return outpath


def plot_dataset_uncertainty_gap(records: List[Record], outdir: Path) -> Optional[Path]:
    per_ds_correct: Dict[str, List[float]] = defaultdict(list)
    per_ds_incorrect: Dict[str, List[float]] = defaultdict(list)
    for r in records:
        if r.u_final is None or r.correct is None:
            continue
        try:
            val = float(r.u_final)
        except Exception:
            continue
        if not math.isfinite(val):
            continue
        key = r.dataset or "<unknown>"
        if r.correct:
            per_ds_correct[key].append(val)
        else:
            per_ds_incorrect[key].append(val)

    gaps: List[Tuple[str, float, float, float]] = []
    for ds in sorted(set(per_ds_correct) | set(per_ds_incorrect)):
        good = per_ds_correct.get(ds, [])
        bad = per_ds_incorrect.get(ds, [])
        if not bad or not good:
            continue
        mean_bad = float(np.mean(bad))
        mean_good = float(np.mean(good))
        gap = mean_bad - mean_good
        gaps.append((ds, gap, mean_bad, mean_good))

    if not gaps:
        return None

    gaps.sort(key=lambda tup: tup[1], reverse=True)
    labels = [g[0] for g in gaps]
    values = [g[1] for g in gaps]

    plt.figure(figsize=(min(12, 0.6 * len(labels) + 4), 5), dpi=150)
    bars = plt.bar(range(len(labels)), values, color="#1f77b4", alpha=0.8)
    plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
    plt.ylabel("Mean u_final(incorrect) - Mean u_final(correct)")
    plt.title("Per-dataset uncertainty gap (higher better)")
    for rect, (_, gap, mean_bad, mean_good) in zip(bars, gaps):
        h = rect.get_height()
        plt.text(
            rect.get_x() + rect.get_width() / 2,
            h + 0.01,
            f"{gap:.2f}\n({mean_bad:.2f}/{mean_good:.2f})",
            ha="center",
            va="bottom",
            fontsize=7,
        )
    plt.tight_layout()
    outpath = outdir / "dataset_uncertainty_gap.png"
    plt.savefig(outpath)
    plt.close()
    return outpath


def plot_token_dynamics(records: List[Record], outdir: Path, max_len: int = 128) -> Tuple[Optional[Path], Optional[Path]]:
    # Aggregate mean u_token[i] across sequences that have index i
    per_idx_vals: List[List[float]] = [[] for _ in range(max_len)]
    per_idx_vals_ok: List[List[float]] = [[] for _ in range(max_len)]
    per_idx_vals_bad: List[List[float]] = [[] for _ in range(max_len)]

    for r in records:
        seq = r.u_token or []
        if not seq:
            continue
        upto = min(len(seq), max_len)
        for i in range(upto):
            v = seq[i]
            if not math.isfinite(v):
                continue
            per_idx_vals[i].append(float(v))
            if r.correct is True:
                per_idx_vals_ok[i].append(float(v))
            elif r.correct is False:
                per_idx_vals_bad[i].append(float(v))

    def _means(vals: List[List[float]]) -> np.ndarray:
        out = np.full(len(vals), np.nan, dtype=float)
        for i, lst in enumerate(vals):
            if lst:
                out[i] = float(np.mean(lst))
        return out

    overall = _means(per_idx_vals)
    ok = _means(per_idx_vals_ok)
    bad = _means(per_idx_vals_bad)

    # Overall line
    overall_path = None
    if np.isfinite(overall).any():
        plt.figure(figsize=(8, 4), dpi=150)
        xs = np.arange(1, len(overall) + 1)
        plt.plot(xs, overall, color="#1f77b4", label="All")
        plt.xlabel("Token index")
        plt.ylabel("u_token mean")
        plt.title("Token-level uncertainty (mean per index)")
        plt.grid(True, linestyle="--", alpha=0.3)
        plt.legend()
        plt.tight_layout()
        overall_path = outdir / "u_token_mean_over_position.png"
        plt.savefig(overall_path)
        plt.close()

    split_path = None
    if np.isfinite(ok).any() or np.isfinite(bad).any():
        plt.figure(figsize=(8, 4), dpi=150)
        xs = np.arange(1, len(ok) + 1)
        if np.isfinite(ok).any():
            plt.plot(xs, ok, color="#2ca02c", label="Correct")
        if np.isfinite(bad).any():
            plt.plot(xs, bad, color="#d62728", label="Incorrect")
        plt.xlabel("Token index")
        plt.ylabel("u_token mean")
        plt.title("Token-level uncertainty by correctness")
        plt.grid(True, linestyle="--", alpha=0.3)
        plt.legend()
        plt.tight_layout()
        split_path = outdir / "u_token_mean_over_position_split.png"
        plt.savefig(split_path)
        plt.close()

    return overall_path, split_path


def plot_spike_positions(records: List[Record], outdir: Path) -> Optional[Path]:
    # Determine a common spike threshold (80th percentile of all token u's)
    all_vals: List[float] = []
    lengths: List[int] = []
    for r in records:
        if r.u_token:
            lengths.append(len(r.u_token))
            for v in r.u_token:
                if math.isfinite(v):
                    all_vals.append(float(v))
    if not all_vals:
        return None
    thr = float(np.quantile(np.array(all_vals), 0.80))

    def first_spike_pos(seq: List[float]) -> Optional[float]:
        if not seq:
            return None
        n = len(seq)
        for i, v in enumerate(seq):
            if math.isfinite(v) and v >= thr:
                return (i + 1) / n  # 0..1 normalized (1-indexed position)
        return None

    ok_pos: List[float] = []
    bad_pos: List[float] = []
    for r in records:
        pos = first_spike_pos(r.u_token or [])
        if pos is None:
            continue
        if r.correct is True:
            ok_pos.append(pos)
        elif r.correct is False:
            bad_pos.append(pos)

    if not ok_pos and not bad_pos:
        return None

    plt.figure(figsize=(7, 4), dpi=150)
    bins = np.linspace(0, 1, 21)
    if ok_pos:
        plt.hist(ok_pos, bins=bins, alpha=0.6, label="Correct", color="#2ca02c", density=True)
    if bad_pos:
        plt.hist(bad_pos, bins=bins, alpha=0.6, label="Incorrect", color="#d62728", density=True)
    plt.xlabel("First spike position (normalized)")
    plt.ylabel("Density")
    plt.title("Where uncertainty spikes first (80th percentile threshold)")
    plt.legend()
    plt.tight_layout()
    outpath = outdir / "u_token_spike_position_hist.png"
    plt.savefig(outpath)
    plt.close()
    return outpath


def plot_length_uncertainty(records: List[Record], outdir: Path) -> Optional[Path]:
    lengths: List[int] = []
    u_vals: List[float] = []
    fail_vals: List[float] = []
    for r in records:
        if r.u_final is None or r.correct is None:
            continue
        if not r.u_token:
            continue
        try:
            u_val = float(r.u_final)
        except Exception:
            continue
        if not math.isfinite(u_val):
            continue
        lengths.append(len(r.u_token))
        u_vals.append(u_val)
        fail_vals.append(1.0 if not r.correct else 0.0)

    if not lengths:
        return None

    len_arr = np.array(lengths, dtype=float)
    u_arr = np.array(u_vals, dtype=float)
    fail_arr = np.array(fail_vals, dtype=float)

    centers_u, mean_u, _ = binned_curve(len_arr, u_arr, bins=10)
    centers_fail, mean_fail, _ = binned_curve(len_arr, fail_arr, bins=10)
    if centers_u.size == 0:
        return None

    plt.figure(figsize=(8, 5), dpi=150)
    plt.plot(centers_u, mean_u, marker="o", color="#1f77b4", label="Mean u_final")
    if centers_fail.size:
        plt.plot(centers_fail, mean_fail, marker="s", color="#d62728", label="Error rate")
    plt.xlabel("Answer length (tokens with u_token)")
    plt.ylabel("Value")
    plt.title("Uncertainty and failure rate vs answer length")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    outpath = outdir / "length_vs_uncertainty.png"
    plt.savefig(outpath)
    plt.close()
    return outpath


def plot_layer_head_dists(records: List[Record], outdir: Path) -> Tuple[Optional[Path], Optional[Path]]:
    layers: List[int] = []
    heads: List[int] = []
    for r in records:
        if r.selected_layer and isinstance(r.selected_layer, str) and r.selected_layer.startswith("l"):
            name = r.selected_layer[1:]
            if name.isdigit():
                layers.append(int(name))
        if r.selected_head is not None:
            try:
                heads.append(int(r.selected_head))
            except Exception:
                pass

    layer_path = head_path = None
    if layers:
        c = Counter(layers)
        xs = sorted(c.keys())
        ys = [c[k] for k in xs]
        plt.figure(figsize=(8, 4), dpi=150)
        plt.bar(xs, ys, color="#1f77b4")
        plt.xlabel("Layer index (best layer)")
        plt.ylabel("Count")
        plt.title("Distribution of best selected layers")
        plt.tight_layout()
        layer_path = outdir / "selected_layer_hist.png"
        plt.savefig(layer_path)
        plt.close()

    if heads:
        c = Counter(heads)
        xs = sorted(c.keys())
        ys = [c[k] for k in xs]
        plt.figure(figsize=(8, 4), dpi=150)
        plt.bar(xs, ys, color="#9467bd")
        plt.xlabel("Head index (best head in best layer)")
        plt.ylabel("Count")
        plt.title("Distribution of best selected heads")
        plt.tight_layout()
        head_path = outdir / "selected_head_hist.png"
        plt.savefig(head_path)
        plt.close()

    return layer_path, head_path


def plot_dataset_auc(records: List[Record], outdir: Path) -> Optional[Path]:
    by_ds: Dict[str, List[Record]] = defaultdict(list)
    for r in records:
        by_ds[r.dataset or "<unknown>"].append(r)

    names: List[str] = []
    aucs: List[float] = []
    for name, recs in sorted(by_ds.items()):
        u = np.array([float(r.u_final) for r in recs if r.u_final is not None and r.correct is not None], dtype=float)
        y = np.array([1 if not r.correct else 0 for r in recs if r.u_final is not None and r.correct is not None], dtype=int)
        if u.size < 2 or np.unique(y).size < 2:
            continue
        (fpr, tpr, auc), _ = roc_pr_from_scores(y, u)
        if not np.isfinite(auc):
            continue
        names.append(name)
        aucs.append(float(auc))
    if not aucs:
        return None
    order = np.argsort(aucs)
    names = [names[i] for i in order]
    aucs = [aucs[i] for i in order]
    plt.figure(figsize=(10, 5), dpi=150)
    bars = plt.bar(range(len(names)), aucs, color="#17becf")
    plt.xticks(range(len(names)), names, rotation=45, ha="right")
    plt.ylim(0, 1)
    plt.ylabel("AUC (incorrect detection via u_final)")
    plt.title("Separability by dataset (RAUQ AUC)")
    for rect, val in zip(bars, aucs):
        h = rect.get_height()
        plt.text(rect.get_x() + rect.get_width() / 2, h + 0.01, f"{val:.2f}", ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    outpath = outdir / "auc_by_dataset.png"
    plt.savefig(outpath)
    plt.close()
    return outpath


def plot_gating_tradeoff(records: List[Record], outdir: Path) -> Optional[Path]:
    # For a range of thresholds on u_final, compute:
    #   - flagged fraction (share of records above threshold)
    #   - error recall among flagged (TPR on incorrect)
    #   - error precision among flagged
    xs = np.array([float(r.u_final) for r in records if r.u_final is not None and r.correct is not None], dtype=float)
    ys = np.array([1 if not r.correct else 0 for r in records if r.u_final is not None and r.correct is not None], dtype=int)
    if xs.size == 0:
        return None
    thresholds = np.quantile(xs, np.linspace(0.0, 0.999, 40))
    flagged_frac: List[float] = []
    error_recall: List[float] = []
    error_precision: List[float] = []
    n_pos = float(np.sum(ys == 1))
    for t in thresholds:
        flagged = xs >= t
        if not np.any(flagged):
            continue
        tp = float(np.sum((ys == 1) & flagged))
        fp = float(np.sum((ys == 0) & flagged))
        flagged_frac.append(float(np.mean(flagged)))
        error_recall.append(tp / n_pos if n_pos > 0 else float("nan"))
        error_precision.append(tp / max(1.0, tp + fp))
    if not flagged_frac:
        return None
    plt.figure(figsize=(7, 5), dpi=150)
    plt.plot(flagged_frac, error_recall, label="Error recall among flagged", color="#d62728")
    plt.plot(flagged_frac, error_precision, label="Error precision among flagged", color="#1f77b4")
    plt.xlabel("Fraction flagged (would trigger CoT)")
    plt.ylabel("Metric")
    plt.title("Gating trade-offs vs fraction flagged (threshold on u_final)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    outpath = outdir / "gating_tradeoffs_vs_flagged_fraction.png"
    plt.savefig(outpath)
    plt.close()
    return outpath


# --------------------------------- Insights --------------------------------


def summarize_insights(records: List[Record], outdir: Path) -> Path:
    # Basic stats
    n = len(records)
    n_with_u = sum(1 for r in records if r.u_final is not None)
    labels = [r.correct for r in records if r.correct is not None]
    acc = accuracy([bool(x) for x in labels]) if labels else float("nan")
    # Macro accuracy (dataset-balanced): average accuracy across datasets
    by_ds_for_macro: Dict[str, List[bool]] = defaultdict(list)
    for r in records:
        if r.correct is None:
            continue
        by_ds_for_macro[r.dataset or "<unknown>"].append(bool(r.correct))
    macro_acc = float(np.mean([accuracy(v) for v in by_ds_for_macro.values()])) if by_ds_for_macro else float("nan")
    # ROC/PR overall for u_final
    xs = np.array([float(r.u_final) for r in records if r.u_final is not None and r.correct is not None], dtype=float)
    ys = np.array([1 if not r.correct else 0 for r in records if r.u_final is not None and r.correct is not None], dtype=int)
    if xs.size >= 2 and np.unique(ys).size >= 2:
        (fpr_u, tpr_u, auc_u), (rec_u, prec_u, ap_u) = roc_pr_from_scores(ys, xs)
    else:
        auc_u = ap_u = float("nan")

    # Dataset AUCs
    by_ds: Dict[str, List[Record]] = defaultdict(list)
    for r in records:
        by_ds[r.dataset or "<unknown>"].append(r)
    auc_by_ds: List[Tuple[str, float]] = []
    for name, recs in by_ds.items():
        x = np.array([float(r.u_final) for r in recs if r.u_final is not None and r.correct is not None], dtype=float)
        y = np.array([1 if not r.correct else 0 for r in recs if r.u_final is not None and r.correct is not None], dtype=int)
        if x.size >= 2 and np.unique(y).size >= 2:
            (fpr, tpr, auc), _ = roc_pr_from_scores(y, x)
            if np.isfinite(auc):
                auc_by_ds.append((name, float(auc)))
    auc_by_ds.sort(key=lambda kv: kv[1])

    # u_final vs max-token gaps
    gaps: List[float] = []
    ratios: List[float] = []
    for r in records:
        if r.u_final is None or not r.u_token:
            continue
        max_u = _max_finite(r.u_token)
        if max_u is None:
            continue
        try:
            uf = float(r.u_final)
        except Exception:
            continue
        if not math.isfinite(uf):
            continue
        gaps.append(max_u - uf)
        if uf > 1e-8:
            ratios.append(max_u / uf)

    avg_gap = float(np.mean(gaps)) if gaps else float("nan")
    p90_gap = float(np.quantile(gaps, 0.9)) if gaps else float("nan")
    avg_ratio = float(np.mean(ratios)) if ratios else float("nan")
    p90_ratio = float(np.quantile(ratios, 0.9)) if ratios else float("nan")

    # Heuristics to highlight weak points
    findings: List[str] = []
    findings.append(f"Records: {n} (with u_final: {n_with_u})")
    findings.append(
        f"Overall accuracy (micro): {acc:.3f}" if np.isfinite(acc) else "Overall accuracy (micro): n/a"
    )
    findings.append(
        f"Overall accuracy (macro, per‑dataset): {macro_acc:.3f}"
        if np.isfinite(macro_acc)
        else "Overall accuracy (macro): n/a"
    )
    findings.append(
        f"RAUQ error-detection AUC: {auc_u:.3f}, AP: {ap_u:.3f}" if np.isfinite(auc_u) else "RAUQ AUC/AP: n/a"
    )
    if auc_by_ds:
        worst = auc_by_ds[0]
        best = auc_by_ds[-1]
        findings.append(f"Worst dataset by separability: {worst[0]} (AUC={worst[1]:.3f})")
        findings.append(f"Best dataset by separability: {best[0]} (AUC={best[1]:.3f})")
        low = [f"{k}={v:.2f}" for k, v in auc_by_ds if v < 0.65]
        if low:
            findings.append("Datasets with weak separation (AUC<0.65): " + ", ".join(low))

    if np.isfinite(avg_gap):
        findings.append(f"Mean max-vs-final gap: {avg_gap:.3f} (p90={p90_gap:.3f})")
    if np.isfinite(avg_ratio):
        findings.append(f"Mean max/final ratio: {avg_ratio:.2f} (p90={p90_ratio:.2f})")

    # Token dynamics quick checks
    token_lens = [len(r.u_token) for r in records if r.u_token]
    if token_lens:
        findings.append(
            f"Token lengths: median={int(np.median(token_lens))}, p90={int(np.quantile(token_lens, 0.9))}"
        )

    # Layer selection checks
    layers = []
    for r in records:
        if r.selected_layer and r.selected_layer.startswith("l") and r.selected_layer[1:].isdigit():
            layers.append(int(r.selected_layer[1:]))
    if layers:
        findings.append(
            f"Best layers: min={min(layers)}, max={max(layers)}, mode={Counter(layers).most_common(1)[0][0]}"
        )

    outpath = outdir / "insights.txt"
    with outpath.open("w", encoding="utf-8") as fh:
        fh.write("RAUQ dashboard insights\n")
        fh.write("======================\n\n")
        for line in findings:
            fh.write(f"- {line}\n")
    return outpath


# ----------------------------------- Main ----------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot a suite of RAUQ diagnostics from JSONL outputs.")
    ap.add_argument(
        "--infile",
        type=Path,
        default=Path("data/artifacts/rauq_output.jsonl"),
        help="Path to rauq_output.jsonl",
    )
    ap.add_argument(
        "--outdir", type=Path, default=Path("data/plots"), help="Directory for output images"
    )
    ap.add_argument(
        "--max_tokens_plot",
        type=int,
        default=128,
        help="Max token positions for token-dynamics plots",
    )
    args = ap.parse_args()

    infile: Path = args.infile
    outdir: Path = args.outdir
    ensure_dir(outdir)

    if not infile.is_file():
        raise SystemExit(f"Input file not found: {infile}")

    records = load_jsonl(infile)
    if not records:
        raise SystemExit("No records parsed from input file.")

    # Summary print
    n_total = len(records)
    n_correct = sum(1 for r in records if r.correct)
    n_labeled = sum(1 for r in records if r.correct is not None)
    n_with_u = sum(1 for r in records if r.u_final is not None)
    print(
        f"Loaded {n_total} records (labeled: {n_labeled}, with u_final: {n_with_u}). Saving plots to {outdir}"
    )

    # Plots
    produced: List[Tuple[str, Optional[Path]]] = []
    produced.append(("acc_by_dataset", plot_acc_by_dataset(records, outdir)))
    produced.append(("u_hist", plot_u_hist_by_correct(records, outdir)))
    produced.append(("max_u_hist", plot_max_u_hist_by_correct(records, outdir)))
    produced.append(("u_vs_max", plot_u_final_vs_max(records, outdir)))
    rpath, ppath = plot_roc_pr(records, outdir)
    produced.append(("roc", rpath))
    produced.append(("pr", ppath))
    produced.append(("binned_u_fail", plot_binned_failure_rate(records, outdir)))
    produced.append(("dataset_uncert_gap", plot_dataset_uncertainty_gap(records, outdir)))
    ovr, split = plot_token_dynamics(records, outdir, max_len=int(args.max_tokens_plot))
    produced.append(("tok_dyn_overall", ovr))
    produced.append(("tok_dyn_split", split))
    produced.append(("spike_pos", plot_spike_positions(records, outdir)))
    lpath, hpath = plot_layer_head_dists(records, outdir)
    produced.append(("layer_hist", lpath))
    produced.append(("head_hist", hpath))
    produced.append(("auc_by_dataset", plot_dataset_auc(records, outdir)))
    produced.append(("gating_tradeoff", plot_gating_tradeoff(records, outdir)))
    produced.append(("length_uncertainty", plot_length_uncertainty(records, outdir)))

    # Insights
    insights_path = summarize_insights(records, outdir)
    produced.append(("insights", insights_path))

    # Print summary of generated artifacts
    made = [name for (name, p) in produced if p is not None]
    missing = [name for (name, p) in produced if p is None]
    print(f"Generated: {', '.join(made)}")
    if missing:
        print(f"Skipped (insufficient data): {', '.join(missing)}")


if __name__ == "__main__":
    main()
