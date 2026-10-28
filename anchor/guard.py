"""Stage 3: target-domain non-inferiority guard.

The guard compares a *frozen* candidate schedule against Native-Full on
independent calibration sources of the deployment domain, averaging
seeds within each source, and deploys the candidate only if a one-sided
source-bootstrap upper bound on the mean difference clears the margin
``delta_NI``. Missing or degenerate calibration abstains (gamma = 1).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .bootstrap import bca_upper_bound


@dataclass
class GuardDecision:
    deploy: bool
    upper_bound: float
    delta_NI: float
    n_sources: int
    reason: str
    gamma: Optional[np.ndarray] = None


def guard_decision(
    candidate_losses: np.ndarray,
    native_losses: np.ndarray,
    sources: Optional[np.ndarray] = None,
    seeds: Optional[np.ndarray] = None,
    delta_NI: float = 0.0,
    level: float = 0.95,
    B: int = 10_000,
    bootstrap_seed: int = 2027,
    min_sources: int = 3,
    candidate_gamma: Optional[np.ndarray] = None,
    native_gamma_is_one: bool = True,
) -> GuardDecision:
    """Decide whether to deploy ``candidate_gamma``.

    Parameters
    ----------
    candidate_losses, native_losses : per-run losses (lower is better),
        aligned elementwise; one entry per (source, seed) run.
    sources : source identifiers; seeds averaged within each source
        before the bootstrap. Defaults to one source per run.
    seeds : optional seed identifiers (kept for manifests).
    delta_NI : non-inferiority margin; the conservative default is 0.
    """
    cand = np.asarray(candidate_losses, dtype=float)
    nat = np.asarray(native_losses, dtype=float)
    if cand.shape != nat.shape:
        raise ValueError("candidate and native losses must align")
    if len(cand) == 0:
        return GuardDecision(False, float("inf"), delta_NI, 0,
                             "no calibration runs", None)
    if sources is None:
        sources = np.arange(len(cand))
    sources = np.asarray(sources)
    if seeds is not None and len(seeds) != len(cand):
        raise ValueError("seeds must align with losses")

    deltas = []
    for s in np.unique(sources):
        m = sources == s
        deltas.append(float((cand[m] - nat[m]).mean()))
    deltas = np.asarray(deltas)
    if len(deltas) < min_sources:
        return GuardDecision(False, float("inf"), delta_NI, len(deltas),
                             f"too few sources ({len(deltas)} < {min_sources})",
                             None)
    ub = bca_upper_bound(deltas, level=level, B=B, seed=bootstrap_seed)
    if ub <= delta_NI:
        return GuardDecision(True, ub, delta_NI, len(deltas),
                             "calibration upper bound clears the margin",
                             candidate_gamma)
    return GuardDecision(False, ub, delta_NI, len(deltas),
                         "calibration upper bound exceeds the margin", None)


def guard_from_manifest(
    rows,
    metric: str = "anchor_lpips",
    arm_candidate: str = "candidate",
    arm_native: str = "native",
    delta_NI: float = 0.0,
    **kwargs,
) -> GuardDecision:
    """Convenience wrapper over a list of dict rows with fields
    ``source``, ``seed``, ``arm``, and ``metric``.
    """
    cand, nat, srcs, seeds = [], [], [], []
    by_key = {}
    for r in rows:
        by_key[(r["source"], r["seed"], r["arm"])] = float(r[metric])
    for (s, sd, arm), val in by_key.items():
        if arm != arm_candidate:
            continue
        natv = by_key.get((s, sd, arm_native))
        if natv is None:
            continue
        cand.append(val)
        nat.append(natv)
        srcs.append(s)
        seeds.append(sd)
    return guard_decision(
        np.asarray(cand), np.asarray(nat),
        sources=np.asarray(srcs), seeds=np.asarray(seeds),
        delta_NI=delta_NI, **kwargs,
    )
