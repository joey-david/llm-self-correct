from __future__ import annotations

import math
from typing import Iterable, Tuple

import numpy as np


def compute_rejection_curve(
    qualities: np.ndarray,
    order: np.ndarray,
    max_remove: int,
) -> np.ndarray:
    """Return average quality of remaining examples after removing top-k entries."""
    if max_remove < 0:
        raise ValueError("max_remove must be non-negative")
    n = qualities.shape[0]
    if n == 0:
        return np.array([], dtype=float)
    if max_remove >= n:
        raise ValueError("max_remove must be less than the number of samples")

    sorted_q = qualities[order]
    total_sum = float(np.sum(sorted_q))
    if not math.isfinite(total_sum):
        raise ValueError("quality contains non-finite values")

    cumsum = np.cumsum(sorted_q)
    curve = np.empty(max_remove + 1, dtype=float)
    for k in range(max_remove + 1):
        removed_sum = float(cumsum[k - 1]) if k > 0 else 0.0
        remaining = n - k
        curve[k] = (total_sum - removed_sum) / max(remaining, 1)
    return curve


def compute_prr(
    qualities: Iterable[float],
    uncertainties: Iterable[float],
    max_fraction: float = 0.5,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Compute Prediction Rejection Ratio (PRR) and return the rejection curve."""
    q = np.asarray(list(qualities), dtype=float)
    u = np.asarray(list(uncertainties), dtype=float)
    if q.size == 0:
        return float("nan"), np.array([], dtype=float), np.array([], dtype=float)

    mask = np.isfinite(q) & np.isfinite(u)
    q = q[mask]
    u = u[mask]
    n = q.size
    if n == 0:
        return float("nan"), np.array([], dtype=float), np.array([], dtype=float)

    max_fraction = float(max_fraction)
    if max_fraction <= 0.0:
        max_remove = 0
    else:
        max_remove = min(int(math.floor(max_fraction * n)), n - 1)

    fractions = np.arange(0, max_remove + 1, dtype=float) / max(1, n)
    if max_remove == 0:
        base_val = float(np.mean(q))
        return 0.0, fractions, np.full_like(fractions, base_val)

    actual_order = np.argsort(-u, kind="mergesort")
    ideal_order = np.argsort(q, kind="mergesort")

    curve_actual = compute_rejection_curve(q, actual_order, max_remove)
    curve_ideal = compute_rejection_curve(q, ideal_order, max_remove)
    base_value = float(np.mean(q))
    base_curve = np.full_like(curve_actual, base_value, dtype=float)

    area_actual = float(np.trapz(curve_actual - base_curve, fractions))
    area_ideal = float(np.trapz(curve_ideal - base_curve, fractions))

    if area_ideal <= 0:
        prr = 0.0
    else:
        prr = max(0.0, min(1.0, area_actual / area_ideal))

    return prr, fractions, curve_actual
