"""Source-level BCa bootstrap used by ANCHOR's statistics.

The unit of resampling is the *source* (clip / scene), not the seeded
run: seeds are averaged within a source before inference, matching the
paper's ``27 sources x 2 seeds`` and ``10+10 sources`` protocols.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from scipy.stats import norm


def _jackknife_acceleration(x: np.ndarray) -> float:
    n = len(x)
    if n < 3:
        return 0.0
    total = x.sum()
    loo = (total - x) / (n - 1)
    d = loo.mean() - loo
    denom = 6.0 * (np.sum(d ** 2) ** 1.5)
    if denom == 0:
        return 0.0
    return float(np.sum(d ** 3) / denom)


def bca_interval(
    x: np.ndarray,
    level: float = 0.95,
    B: int = 10_000,
    seed: int = 2027,
) -> Tuple[float, float, float]:
    """Two-sided BCa interval for the mean of ``x`` (source-level).

    Returns ``(lo, hi, point)``. With fewer than three sources the
    interval degrades gracefully to the percentile interval.
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n == 0:
        raise ValueError("empty sample")
    point = float(x.mean())
    if n == 1:
        return point, point, point

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(B, n))
    boot = x[idx].mean(axis=1)
    alpha = 1.0 - level
    z0 = norm.ppf(np.clip(np.mean(boot < point), 1e-6, 1 - 1e-6))
    acc = _jackknife_acceleration(x)
    out = []
    for q in (alpha / 2.0, 1.0 - alpha / 2.0):
        zq = norm.ppf(q)
        denom = 1.0 - acc * (z0 + zq)
        if abs(denom) < 1e-12:
            adj = q
        else:
            adj = norm.cdf(z0 + (z0 + zq) / denom)
        out.append(float(np.quantile(boot, np.clip(adj, 1e-4, 1 - 1e-4))))
    return out[0], out[1], point


def bca_upper_bound(
    x: np.ndarray,
    level: float = 0.95,
    B: int = 10_000,
    seed: int = 2027,
) -> float:
    """One-sided ``level`` upper confidence bound for ``E[x]``.

    The guard uses this with ``level = 1 - eta`` and compares against
    the non-inferiority margin ``delta_NI``.
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n == 0:
        raise ValueError("empty sample")
    if n == 1:
        return float(x[0])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(B, n))
    boot = x[idx].mean(axis=1)
    z0 = norm.ppf(np.clip(np.mean(boot < x.mean()), 1e-6, 1 - 1e-6))
    acc = _jackknife_acceleration(x)
    zq = norm.ppf(level)
    denom = 1.0 - acc * (z0 + zq)
    adj = level if abs(denom) < 1e-12 else norm.cdf(z0 + (z0 + zq) / denom)
    return float(np.quantile(boot, np.clip(adj, 1e-4, 1 - 1e-4)))


def cluster_bootstrap_interval(
    x: np.ndarray,
    clusters: np.ndarray,
    level: float = 0.95,
    B: int = 10_000,
    seed: int = 2027,
) -> Tuple[float, float]:
    """Cluster bootstrap: resample clusters, average within cluster first.

    Used for the aggregate log-ratio audit (clusters = run groups).
    """
    x = np.asarray(x, dtype=float)
    clusters = np.asarray(clusters)
    uniq = np.unique(clusters)
    means = np.array([x[clusters == c].mean() for c in uniq])
    lo, hi, _ = bca_interval(means, level=level, B=B, seed=seed)
    return lo, hi
