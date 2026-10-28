"""Stage 2: confidence-aware classification (Wall / Persistent /
Droppable / Uncertain).

Implements Eq. (3) of the paper. All bounds are one-sided
source-bootstrap bounds; the noise floor is ``h = 3 * sigma_probe^2``.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np

from .bootstrap import bca_interval


class Regime(str, Enum):
    WALL = "Wall"
    PERSISTENT = "Persistent"
    DROPPABLE = "Droppable"
    UNCERTAIN = "Uncertain"


@dataclass
class ClassificationResult:
    regime: Regime
    R_lo: float
    R_hi: float
    S_lo: Optional[float]
    S_hi: Optional[float]
    h: float
    reason: str


def classify_regime(
    R: np.ndarray,
    S: Optional[np.ndarray] = None,
    sigma_probe: float = 0.0,
    level: float = 0.95,
    B: int = 10_000,
    seed: int = 2027,
    min_sources: int = 3,
) -> ClassificationResult:
    """Classify an interface--domain pair from per-source statistics.

    Parameters
    ----------
    R : per-source reachability mass (R = sum_s w_s ||G_s||^2)
    S : per-source signed alignment, or None if the probe cannot
        estimate the signed component (then only Wall/Uncertain apply)
    sigma_probe : probe-noise standard deviation (null-swap floor)
    """
    R = np.asarray(R, dtype=float)
    h = 3.0 * float(sigma_probe) ** 2

    if len(R) < min_sources:
        return ClassificationResult(
            Regime.UNCERTAIN, float("nan"), float("nan"),
            None, None, h,
            f"too few sources ({len(R)} < {min_sources})",
        )

    r_lo, r_hi, _ = bca_interval(R, level=level, B=B, seed=seed)
    if r_hi < h:
        return ClassificationResult(
            Regime.WALL, r_lo, r_hi, None, None, h,
            "R upper bound below the null-swap floor",
        )

    if S is None:
        return ClassificationResult(
            Regime.UNCERTAIN, r_lo, r_hi, None, None, h,
            "reachability present but signed component unavailable",
        )
    S = np.asarray(S, dtype=float)
    if len(S) != len(R):
        raise ValueError("R and S must be per-source with equal length")
    s_lo, s_hi, _ = bca_interval(S, level=level, B=B, seed=seed)

    if r_lo >= h and s_lo > 0:
        return ClassificationResult(
            Regime.PERSISTENT, r_lo, r_hi, s_lo, s_hi, h,
            "reachable and early alignment positive at the lower bound",
        )
    if r_lo >= h and s_hi <= 0:
        return ClassificationResult(
            Regime.DROPPABLE, r_lo, r_hi, s_lo, s_hi, h,
            "reachable and early alignment non-positive at the upper bound",
        )
    return ClassificationResult(
        Regime.UNCERTAIN, r_lo, r_hi, s_lo, s_hi, h,
        "confidence bounds do not separate a regime",
    )


def classify_from_statistics(
    R: float,
    R_lo: float,
    R_hi: float,
    S: Optional[float] = None,
    S_lo: Optional[float] = None,
    S_hi: Optional[float] = None,
    sigma_probe: float = 0.0,
) -> Regime:
    """Apply Eq. (3) directly to precomputed bounds.

    Useful when the bounds come from an external bootstrap run whose
    per-source values are not available in this process.
    """
    h = 3.0 * float(sigma_probe) ** 2
    if R_hi < h:
        return Regime.WALL
    if R_lo >= h:
        if S is not None and S_lo is not None and S_lo > 0:
            return Regime.PERSISTENT
        if S is not None and S_hi is not None and S_hi <= 0:
            return Regime.DROPPABLE
    return Regime.UNCERTAIN
