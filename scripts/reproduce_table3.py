#!/usr/bin/env python3
"""Reproduce the DAVIS 10+10 locked-lane table (paper Table III / IV).

Input CSV columns: source, split, seed, arm, anchor_lpips
  split ∈ {train, val}; arm ∈ {native, candidate}
  Seeds are averaged within each source before inference (paper rule).

Output: JSON with the train promotion gate and the val evaluation,
including source-level deltas, BCa intervals, W/T/L, and the guard
decision at delta_NI = 0.

Usage:
  python scripts/reproduce_table3.py --csv examples/davis_lane_example.csv
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

from anchor.bootstrap import bca_interval, bca_upper_bound  # noqa: E402
from anchor.guard import guard_decision  # noqa: E402


def load(path):
    rows = defaultdict(list)
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            rows[(r["source"], r["split"], r["arm"])].append(
                float(r["anchor_lpips"])
            )
    return rows


def source_deltas(rows, split):
    srcs = sorted({k[0] for k in rows if k[1] == split})
    out = []
    for s in srcs:
        cand = rows.get((s, split, "candidate"))
        nat = rows.get((s, split, "native"))
        if not cand or not nat:
            continue
        out.append((s, float(np.mean(cand)) - float(np.mean(nat)),
                    float(np.mean(cand)), float(np.mean(nat))))
    return out


def summarize(rows, split, level=0.95, B=10_000, seed=2027):
    d = source_deltas(rows, split)
    if not d:
        return None
    deltas = np.array([x[1] for x in d])
    lo, hi, point = bca_interval(deltas, level=level, B=B, seed=seed)
    w = int(np.sum(deltas < 0))
    t = int(np.sum(deltas == 0))
    l = int(np.sum(deltas > 0))
    return {
        "split": split,
        "n_sources": len(d),
        "native_mean": float(np.mean([x[3] for x in d])),
        "candidate_mean": float(np.mean([x[2] for x in d])),
        "mean_delta": point,
        "bca": [lo, hi],
        "wins_ties_losses": [w, t, l],
        "per_source": [
            {"source": s, "delta": dd, "candidate": c, "native": n}
            for s, dd, c, n in d
        ],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", default="runs/table3_davis.json")
    ap.add_argument("--B", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=2027)
    args = ap.parse_args()

    rows = load(args.csv)
    train = summarize(rows, "train", B=args.B, seed=args.seed)
    val = summarize(rows, "val", B=args.B, seed=args.seed)
    gate = None
    if train:
        d_tr = np.array([x["delta"] for x in train["per_source"]])
        ub = bca_upper_bound(d_tr, level=0.95, B=args.B, seed=args.seed)
        gate = {
            "upper_bound": ub,
            "delta_NI": 0.0,
            "passed": bool(ub <= 0.0),
        }
    out = {"train": train, "val": val, "promotion_gate": gate}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
