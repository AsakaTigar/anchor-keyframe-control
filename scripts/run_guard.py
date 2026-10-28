#!/usr/bin/env python3
"""Apply the target-domain non-inferiority guard to a calibration CSV.

Input CSV columns: source, seed, arm, anchor_lpips (or --metric).

Usage:
  python scripts/run_guard.py --csv examples/davis_lane_train_example.csv \
      --candidate candidate --native native --delta-NI 0
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from anchor.guard import guard_from_manifest  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--metric", default="anchor_lpips")
    ap.add_argument("--candidate", default="candidate")
    ap.add_argument("--native", default="native")
    ap.add_argument("--delta-NI", type=float, default=0.0)
    ap.add_argument("--level", type=float, default=0.95)
    ap.add_argument("--B", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=2027)
    ap.add_argument("--out", default="runs/guard_decision.json")
    args = ap.parse_args()

    with open(args.csv, newline="") as f:
        rows = list(csv.DictReader(f))
    dec = guard_from_manifest(
        rows,
        metric=args.metric,
        arm_candidate=args.candidate,
        arm_native=args.native,
        delta_NI=args.delta_NI,
        level=args.level,
        B=args.B,
        bootstrap_seed=args.seed,
    )
    out = {
        "deploy": dec.deploy,
        "upper_bound": dec.upper_bound,
        "delta_NI": dec.delta_NI,
        "n_sources": dec.n_sources,
        "reason": dec.reason,
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
