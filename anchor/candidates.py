"""Candidate schedules for the three regimes.

Schedules are vectors over execution order (``gamma_s`` multiplies the
native condition at reverse step ``s``; ``1`` = full native condition,
``0`` = null path). The paper's conventions:

* Native-Full   : gamma = 1
* Persistent    : micro-skip K=1, i.e. ``1[s > 1]``
* Droppable     : ``sigmoid(alpha * (logSNR_s - tau))`` with
                  ``(alpha, tau)`` frozen before target-domain evaluation
* Hard skip-K   : ``1[s > K]`` (the alpha -> infinity limit)
"""

from __future__ import annotations

from typing import Optional

import numpy as np


def native_full(T: int) -> np.ndarray:
    return np.ones(T, dtype=float)


def skip_k(T: int, K: int) -> np.ndarray:
    """Hard skip-K: the first K execution-order steps use gamma=0."""
    if not (0 <= K < T):
        raise ValueError("K must satisfy 0 <= K < T")
    g = np.ones(T, dtype=float)
    g[:K] = 0.0
    return g


def micro_skip(T: int) -> np.ndarray:
    return skip_k(T, 1)


def logsnr_schedule(
    alpha_bar: np.ndarray,
    tau: float = -4.495,
    alpha: float = 4.0,
) -> np.ndarray:
    """Droppable candidate from per-step cumulative alpha bars.

    ``alpha_bar`` must be indexed in execution order (the same order as
    the schedule), typically the sampler's ``alphas_cumprod`` sequence.
    """
    a = np.clip(np.asarray(alpha_bar, dtype=float), 1e-12, 1 - 1e-12)
    logsnr = np.log(a) - np.log1p(-a)
    return 1.0 / (1.0 + np.exp(-alpha * (logsnr - tau)))


def logsnr_sigmoid(
    logsnr: np.ndarray,
    tau: float = -4.495,
    alpha: float = 4.0,
) -> np.ndarray:
    """Same as :func:`logsnr_schedule` but takes logSNR directly."""
    logsnr = np.asarray(logsnr, dtype=float)
    return 1.0 / (1.0 + np.exp(-alpha * (logsnr - tau)))


def regime_schedule(
    regime: str,
    T: int,
    alpha_bar: Optional[np.ndarray] = None,
    tau: float = -4.495,
    alpha: float = 4.0,
) -> np.ndarray:
    """Map a regime name to its proposed schedule (Eq. 4)."""
    regime = regime.lower()
    if regime in {"wall", "uncertain"}:
        return native_full(T)
    if regime == "persistent":
        return micro_skip(T)
    if regime == "droppable":
        if alpha_bar is None:
            raise ValueError("droppable candidate needs alpha_bar")
        return logsnr_schedule(alpha_bar, tau=tau, alpha=alpha)
    raise ValueError(f"unknown regime: {regime}")
