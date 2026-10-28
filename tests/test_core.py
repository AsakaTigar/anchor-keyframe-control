"""Unit tests for the ANCHOR reference implementation (no GPU needed).

Run with:  python -m pytest -q tests/
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from anchor import (  # noqa: E402
    ProbeDesign,
    Regime,
    bca_interval,
    classify_regime,
    guard_decision,
    native_full,
    ridge_operator,
    skip_k,
    summarize_response,
)
from anchor.candidates import logsnr_schedule  # noqa: E402
from anchor.classify import classify_from_statistics  # noqa: E402
from anchor.probe import early_alignment, stability_bound  # noqa: E402
from anchor.bootstrap import bca_upper_bound  # noqa: E402


def _smooth_operator(T=12, d=6, rng=None):
    rng = rng or np.random.default_rng(0)
    base = rng.normal(size=(d, 1))
    prof = np.exp(-0.5 * ((np.arange(T) - T / 3) / 3.0) ** 2)
    return base * prof[None, :]


def test_probe_design_shapes_and_budget():
    des = ProbeDesign(T=12, m=10, eps=0.1, seed=1)
    V = des.V()
    assert V.shape == (12, 10)
    assert set(np.unique(V)) <= {-1.0, 1.0}
    assert des.forward_pass_count() == 20
    g = np.ones(12)
    schedules = list(des.schedules(g))
    assert len(schedules) == 10
    for gp, gm in schedules:
        assert gp.shape == gm.shape == (12,)
        assert np.all(gp >= 0) and np.all(gp <= 1)


def test_ridge_recovers_smooth_operator():
    rng = np.random.default_rng(3)
    T, d, m = 12, 6, 24
    G_true = _smooth_operator(T, d, rng)
    des = ProbeDesign(T=T, m=m, eps=0.1, seed=7)
    V = des.V()
    Y = G_true @ V + rng.normal(0, 1e-3, size=(d, m))
    G_hat = ridge_operator(Y, V, eps=0.1, mu=1e-2)
    rel = np.linalg.norm(G_hat - G_true) / np.linalg.norm(G_true)
    assert rel < 0.2, f"relative recovery error too large: {rel}"


def test_summary_and_alignment():
    rng = np.random.default_rng(4)
    G = _smooth_operator(12, 3, rng)
    s = summarize_response(G)
    assert s["R"] > 0
    e0 = np.array([1.0, 0.5, -0.2])
    S = early_alignment(G, e0)
    assert np.isfinite(S)


def test_classifier_quadrants():
    # Wall: tiny R with a large floor
    r = np.full(6, 0.01)
    res = classify_regime(r, np.zeros(6), sigma_probe=1.0)
    assert res.regime is Regime.WALL
    # Persistent: large R, positive S
    r = np.full(6, 1.0) + np.linspace(-0.02, 0.02, 6)
    s = np.full(6, 0.5) + np.linspace(-0.01, 0.01, 6)
    res = classify_regime(r, s, sigma_probe=0.01)
    assert res.regime is Regime.PERSISTENT
    # Droppable: large R, non-positive S
    s = np.full(6, -0.5) + np.linspace(-0.01, 0.01, 6)
    res = classify_regime(r, s, sigma_probe=0.01)
    assert res.regime is Regime.DROPPABLE
    # Uncertain: S interval crosses zero
    s = np.linspace(-1.0, 1.0, 6)
    res = classify_regime(r, s, sigma_probe=0.01)
    assert res.regime is Regime.UNCERTAIN
    # direct Eq. (3) path
    assert classify_from_statistics(1.0, 0.9, 1.1, 0.5, 0.1, 0.9,
                                    sigma_probe=0.01) is Regime.PERSISTENT
    assert classify_from_statistics(1.0, 0.9, 1.1, None,
                                    sigma_probe=0.01) is Regime.UNCERTAIN


def test_guard_deploy_and_abstain():
    rng = np.random.default_rng(5)
    src = np.repeat(np.arange(10), 2)
    seeds = np.tile([42, 123], 10)
    native = rng.normal(0.5, 0.02, size=20)
    better = native - 0.15 + rng.normal(0, 0.01, size=20)
    dec = guard_decision(better, native, sources=src, seeds=seeds)
    assert dec.deploy and dec.upper_bound <= 0.0
    worse = native + 0.2 + rng.normal(0, 0.01, size=20)
    dec = guard_decision(worse, native, sources=src, seeds=seeds)
    assert not dec.deploy
    # too few sources abstains
    dec = guard_decision(better[:2], native[:2])
    assert not dec.deploy


def test_bca_behaviour():
    rng = np.random.default_rng(6)
    x = rng.normal(0.0, 1.0, size=40)
    lo, hi, pt = bca_interval(x, B=2000, seed=1)
    assert lo < pt < hi
    ub = bca_upper_bound(x, level=0.95, B=2000, seed=1)
    assert ub > pt
    # broad shift should push the upper bound above zero
    y = x + 5.0
    assert bca_upper_bound(y, B=2000, seed=1) > 0.0


def test_candidates():
    T = 10
    g = native_full(T)
    assert np.all(g == 1)
    g = skip_k(T, 3)
    assert np.all(g[:3] == 0) and np.all(g[3:] == 1)
    # alpha_bar decreasing from 1 to 0 yields a decreasing sigmoid
    ab = np.linspace(0.999, 0.01, T)
    g = logsnr_schedule(ab, tau=-4.495, alpha=4.0)
    assert g.shape == (T,)
    assert np.all(g >= 0) and np.all(g <= 1)
    d = np.diff(g)
    assert np.all(d <= 1e-9) or np.all(d >= -1e-9)  # monotone either way
    # high logSNR (early) -> gamma near 1, low logSNR (late) -> near 0
    assert g[0] > g[-1]


def test_stability_bound():
    assert stability_bound(0.1, 0.1, 1.0, mu=1.0) == 0.2 / 0.9
    assert stability_bound(0.1, 2.0, 1.0, mu=1.0) == float("inf")
