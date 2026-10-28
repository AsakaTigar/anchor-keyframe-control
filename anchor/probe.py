"""Signed paired probe and ridge recovery of the response operators.

Implements Stage 1 of ANCHOR:

* Rademacher designs ``v_j in {-1, +1}^T`` perturb a reference schedule
  ``gamma_bar`` symmetrically: ``x_j^+-=x(gamma_bar +- eps * v_j)``.
* Paired differences cancel the shared stochastic trajectory:
  ``Y_j = (phi(x_j^+) - phi(x_j^-)) / (2 eps)``.
* Ridge regression with a second-difference penalty recovers the
  target operator ``G`` (and, with an off-target feature map, the
  leakage operator ``L``):
  ``G_hat = Y V^T (V V^T + mu D2^T D2)^(-1)``.

All arrays are plain ``numpy``; no torch dependency.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np


def second_difference_matrix(T: int) -> np.ndarray:
    """(T-2, T) second-difference operator D2 (rows sum to zero)."""
    if T < 3:
        raise ValueError("second-difference penalty needs T >= 3")
    D = np.zeros((T - 2, T))
    for i in range(T - 2):
        D[i, i] = 1.0
        D[i, i + 1] = -2.0
        D[i, i + 2] = 1.0
    return D


@dataclass
class ProbeDesign:
    """A frozen Rademacher design.

    Attributes
    ----------
    T : number of reverse steps
    m : number of probe pairs (each pair costs 2 forward passes)
    eps : perturbation radius in schedule space
    seed : design seed (recorded in the run manifest)
    """

    T: int
    m: int
    eps: float
    seed: int

    def __post_init__(self) -> None:
        if self.T <= 0 or self.m <= 0:
            raise ValueError("T and m must be positive")
        if not (0.0 < self.eps <= 0.5):
            raise ValueError("eps must lie in (0, 0.5]")

    def V(self) -> np.ndarray:
        """(T, m) Rademacher design, deterministic in ``seed``."""
        rng = np.random.default_rng(self.seed)
        return rng.choice([-1.0, 1.0], size=(self.T, self.m))

    def schedules(self, gamma_bar: np.ndarray) -> Iterable[np.ndarray]:
        """Yield ``(gamma_plus, gamma_minus)`` for each probe pair."""
        V = self.V()
        for j in range(self.m):
            yield (
                np.clip(gamma_bar + self.eps * V[:, j], 0.0, 1.0),
                np.clip(gamma_bar - self.eps * V[:, j], 0.0, 1.0),
            )

    def forward_pass_count(self) -> int:
        """Total backbone forward passes: two per probe pair."""
        return 2 * self.m


def ridge_operator(
    Y: np.ndarray,
    V: np.ndarray,
    eps: float,
    mu: float = 1e-2,
    D2: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Recover ``G_hat`` (d, T) from paired differences.

    Parameters
    ----------
    Y : (d, m) paired feature differences divided by ``2 eps``
    V : (T, m) Rademacher design actually used
    eps : perturbation radius (kept for symmetry checks, unused)
    mu : ridge weight on the second-difference penalty
    D2 : optional (T-2, T) penalty; default is the second difference
    """
    del eps  # already folded into Y by the caller
    T = V.shape[0]
    if D2 is None:
        D2 = second_difference_matrix(T) if T >= 3 else np.zeros((0, T))
    A = V @ V.T + mu * (D2.T @ D2 if D2.size else 0.0)
    # Solve A^T G^T = V Y^T  (A symmetric, so A^T = A)
    Gt = np.linalg.solve(A, V @ Y.T)  # (T, d)
    return Gt.T


def summarize_response(
    G: np.ndarray,
    L: Optional[np.ndarray] = None,
    weights: Optional[np.ndarray] = None,
) -> dict:
    """Compute R (reachability mass), U (leakage energy), and norms.

    ``G`` is (d, T) or (T,) with per-step operator columns; ``L`` has
    the same shape when available. ``R = sum_s w_s ||G_s||^2`` and
    ``U = sum_s w_s ||L_s||^2`` with uniform weights by default.
    """
    G = np.atleast_2d(G)
    T = G.shape[-1]
    if weights is None:
        weights = np.full(T, 1.0 / T)
    w = np.asarray(weights, dtype=float)
    r2 = np.sum(G ** 2, axis=0) if G.ndim == 2 else G ** 2
    R = float(np.sum(w * r2))
    out = {"R": R, "r": np.sqrt(r2)}
    if L is not None:
        L = np.atleast_2d(L)
        l2 = np.sum(L ** 2, axis=0) if L.ndim == 2 else L ** 2
        out["U"] = float(np.sum(w * l2))
        out["u"] = np.sqrt(l2)
    return out


def early_alignment(G: np.ndarray, e0: np.ndarray, K0: Optional[int] = None) -> float:
    """Signed alignment ``S = <g_early, e0>`` with ``K0 = ceil(T/6)``."""
    G = np.atleast_2d(G)
    T = G.shape[-1]
    if K0 is None:
        K0 = int(np.ceil(T / 6.0))
    g_early = G[:, :K0].sum(axis=1)
    return float(np.dot(g_early, e0))


def stability_bound(
    delta_b: float,
    delta_Q: float,
    gamma_star_norm: float,
    mu: float,
) -> float:
    """Theorem (calibration stability) right-hand side."""
    denom = mu - delta_Q
    if denom <= 0:
        return float("inf")
    return (delta_b + delta_Q * gamma_star_norm) / denom
