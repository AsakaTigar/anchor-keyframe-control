#!/usr/bin/env python3
"""Reproduce the SSv2 source-level table (paper Table II).

Input CSV columns: source, seed, arm, anchor_lpips
  arms: native, fixed_gate, skipk, ras   (rename via --arms)

Outputs per-arm mean anchor LPIPS, source-level deltas vs native with
BCa intervals, W/T/L over seeded runs, and sign-flip permutation p-values
with Holm correction across arms.

Usage:
  python scripts/reproduce_table2.py --csv examples/ssv2_example.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from anchor.bootstrap import bca_interval  # noqa: E402


def load(path):
    runs = defaultdict(dict)     # (source, seed) -> arm -> value
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            runs[(r["source"], int(r["seed"]))][r["arm"]] = float(
                r["anchor_lpips"]
            )
    return runs


def sign_flip_p(deltas: np.ndarray, B: int = 20_000, seed: int = 2027) -> float:
    """Two-sided sign-flip permutation test for the mean of deltas."""
    rng = np.random.default_rng(seed)
    obs = abs(deltas.mean())
    n = len(deltas)
    signs = rng.choice([-1.0, 1.0], size=(B, n))
    perm = np.abs((signs * deltas).mean(axis=1))
    return float((np.sum(perm >= obs) + 1) / (B + 1))


def holm(pvals):
    order = np.argsort(pvals)
    m = len(pvals)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        val = min(1.0, (m - rank) * pvals[idx])
        running = max(running, val)
        adj[idx] = running
    return adj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", default="runs/table2_ssv2.json")
    ap.add_argument("--arms", nargs="+",
                    default=["fixed_gate", "skipk", "ras"])
    ap.add_argument("--native", default="native")
    ap.add_argument("--B", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=2027)
    args = ap.parse_args()

    runs = load(args.csv)
    sources = sorted({k[0] for k in runs})
    result = {"n_sources": len(sources), "n_runs": len(runs), "arms": {}}

    native_by_src = defaultdict(list)
    for (s, sd), arms in runs.items():
        if args.native in arms:
            native_by_src[s].append(arms[args.native])
    native_mean = {s: float(np.mean(v)) for s, v in native_by_src.items()}
    result["native_mean_anchor_lpips"] = float(np.mean(list(native_mean.values())))

    raw_p = []
    for arm in args.arms:
        per_src_delta = []
        wtl = [0, 0, 0]
        arm_vals = []
        for (s, sd), arms in runs.items():
            if arm not in arms or args.native not in arms:
                continue
            arm_vals.append(arms[arm])
        for s in sources:
            cand = [arms[arm] for (src, sd), arms in runs.items()
                    if src == s and arm in arms]
            nat = [arms[args.native] for (src, sd), arms in runs.items()
                   if src == s and args.native in arms]
            if not cand or not nat:
                continue
            d = float(np.mean(cand) - np.mean(nat))
            per_src_delta.append(d)
            if d < 0:
                wtl[0] += 1
            elif d == 0:
                wtl[1] += 1
            else:
                wtl[2] += 1
        deltas = np.asarray(per_src_delta)
        lo, hi, point = bca_interval(deltas, B=args.B, seed=args.seed)
        p = sign_flip_p(deltas, seed=args.seed)
        raw_p.append(p)
        result["arms"][arm] = {
            "mean_anchor_lpips": float(np.mean(arm_vals)) if arm_vals else None,
            "mean_delta": point,
            "bca": [lo, hi],
            "wins_ties_losses": wtl,
            "p_signflip": p,
        }
    if raw_p:
        adj = holm(np.asarray(raw_p))
        for arm, pa in zip(args.arms, adj):
            result["arms"][arm]["p_holm"] = float(pa)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
