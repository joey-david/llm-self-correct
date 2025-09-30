import argparse
import json
import math
import sys
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np
import matplotlib

#!/usr/bin/env python3


# Use a non-interactive backend for headless environments
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_xy(filepath: Path) -> Tuple[np.ndarray, np.ndarray]:
    xs: List[float] = []
    ys: List[float] = []
    with filepath.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except json.JSONDecodeError:
                # Skip lines that aren't valid JSON
                continue
            x = obj.get("alignscore_best", None)
            y = obj.get("u_final", None)
            if x is None or y is None:
                continue
            try:
                xf = float(x)
                yf = float(y)
                if math.isfinite(xf) and math.isfinite(yf):
                    xs.append(xf)
                    ys.append(yf)
            except (TypeError, ValueError):
                continue
    if not xs:
        raise ValueError("No valid (alignscore_best, u_final) pairs found.")
    return np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)


def kfold_indices(n: int, k: int, seed: int = 0) -> List[np.ndarray]:
    k = max(2, min(k, n))  # at least 2 folds, at most n
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    folds = np.array_split(idx, k)
    return folds


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def select_degree_cv(
    x: np.ndarray,
    y: np.ndarray,
    degrees: List[int],
    kfolds: int,
    seed: int,
) -> int:
    n = x.shape[0]
    folds = kfold_indices(n, kfolds, seed)
    best_deg = degrees[0]
    best_score = float("inf")
    for deg in degrees:
        fold_scores: List[float] = []
        for i in range(len(folds)):
            val_idx = folds[i]
            train_idx = np.concatenate([folds[j] for j in range(len(folds)) if j != i])
            x_train, y_train = x[train_idx], y[train_idx]
            x_val, y_val = x[val_idx], y[val_idx]
            # Fit polynomial
            try:
                coeffs = np.polyfit(x_train, y_train, deg=deg)
            except np.linalg.LinAlgError:
                continue
            y_pred = np.polyval(coeffs, x_val)
            fold_scores.append(rmse(y_val, y_pred))
        if fold_scores:
            score = float(np.mean(fold_scores))
            if score < best_score:
                best_score = score
                best_deg = deg
    return best_deg


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot <= 0:
        return 1.0 if ss_res <= 0 else 0.0
    return 1.0 - (ss_res / ss_tot)


def plot_and_save(
    x: np.ndarray,
    y: np.ndarray,
    coeffs: np.ndarray,
    outpath: Path,
    degree: int,
    title_prefix: str = "",
):
    # Scatter
    plt.figure(figsize=(7, 5), dpi=150)
    plt.scatter(x, y, s=12, alpha=0.6, color="#1f77b4", edgecolors="none", label="Data")

    # Fit line (smooth)
    x_min, x_max = float(np.min(x)), float(np.max(x))
    x_pad = 0.02 * (x_max - x_min if x_max > x_min else 1.0)
    xs = np.linspace(x_min - x_pad, x_max + x_pad, 400)
    ys = np.polyval(coeffs, xs)

    # Compute R^2 on observed data
    y_fit = np.polyval(coeffs, x)
    r2 = r2_score(y, y_fit)

    label = f"Poly deg={degree}, R^2={r2:.3f}"
    plt.plot(xs, ys, color="#d62728", linewidth=2.0, label=label)

    plt.xlabel("alignscore_best")
    plt.ylabel("u_final")
    title = f"{title_prefix}u_final vs alignscore_best"
    if degree is not None:
        title += f" (deg {degree})"
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outpath)
    plt.close()


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Plot u_final vs alignscore_best and fit a simple polynomial."
    )
    parser.add_argument("filepath", type=Path, help="Path to JSONL file.")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output image path (e.g., plot.png). Defaults next to input.",
    )
    parser.add_argument(
        "--degree",
        type=int,
        default=None,
        help="Force polynomial degree (e.g., 1, 2, or 3). If not set, chosen via simple CV.",
    )
    parser.add_argument(
        "--max-degree",
        type=int,
        default=3,
        help="Max degree to consider if auto-selecting (1..max-degree).",
    )
    parser.add_argument(
        "--folds",
        type=int,
        default=5,
        help="Number of folds for CV when auto-selecting degree.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for CV shuffling.",
    )
    args = parser.parse_args(argv)

    x, y = load_xy(args.filepath)

    # Determine output path
    if args.out is not None:
        outpath = args.out
    else:
        stem = args.filepath.stem
        outpath = args.filepath.with_name(f"{stem}_u_final_vs_alignscore.png")

    # Choose degree
    if args.degree is not None:
        degree = max(1, int(args.degree))
    else:
        degrees = list(range(1, max(1, args.max_degree) + 1))
        degree = select_degree_cv(x, y, degrees=degrees, kfolds=args.folds, seed=args.seed)

    # Fit on all data
    coeffs = np.polyfit(x, y, deg=degree)

    # Plot and save
    title_prefix = ""
    plot_and_save(x, y, coeffs, outpath, degree, title_prefix=title_prefix)

    print(f"Saved figure to: {outpath}")
    return 0


if __name__ == "__main__":
    sys.exit(main())