#!/usr/bin/env python3
"""Run the ANCHOR signed paired probe on one interface--domain pair.

Examples
--------
# Synthetic smoke test (no GPU, no weights)
python scripts/run_probe.py --backbone synthetic --sources 6 --m 16 --T 12

# CogVideoX (requires COGVIDEOX_I2V_DIR and weights)
python scripts/run_probe.py --backbone cogvideox --sources 5 --m 24 \
    --anchors data/davis_first_frames --out runs/probe_davis.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from anchor import (  # noqa: E402
    ProbeDesign,
    classify_regime,
    ridge_operator,
    summarize_response,
)
from anchor.probe import early_alignment  # noqa: E402


def features(frames, p: int) -> np.ndarray:
    """Toy feature map: target mean + off-target mean (6 dims)."""
    x = np.asarray(frames, dtype=float)
    if x.max() > 1.5:
        x /= 255.0
    tgt = x[p].mean(axis=(0, 1))
    off = np.delete(x, p, axis=0).mean(axis=(0, 1, 2))
    return np.concatenate([tgt, off])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="synthetic",
                    choices=["synthetic", "cogvideox"])
    ap.add_argument("--sources", type=int, default=6)
    ap.add_argument("--m", type=int, default=16, help="probe pairs")
    ap.add_argument("--eps", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=2027)
    ap.add_argument("--T", type=int, default=12)
    ap.add_argument("--p", type=int, default=6)
    ap.add_argument("--out", default="runs/probe.json")
    args = ap.parse_args()

    if args.backbone == "cogvideox":
        from anchor.backbones import CogVideoXAdapter

        bb = CogVideoXAdapter(steps=args.T)
    else:
        from anchor.backbones import SyntheticBackbone

        bb = SyntheticBackbone(T=args.T)

    rng = np.random.default_rng(args.seed)
    design = ProbeDesign(T=args.T, m=args.m, eps=args.eps, seed=args.seed)
    gamma_bar = np.ones(args.T)
    V = design.V()

    R_vals, S_vals, per_source = [], [], []
    for i in range(args.sources):
        a_lo = rng.random((24, 24, 3))
        a_hi = np.clip(a_lo + 0.25, 0, 1)
        cols = []
        for j, (gp, gm) in enumerate(design.schedules(gamma_bar)):
            xp = features(bb.generate(a_hi, "demo", args.p, 1000 + i, gp), args.p)
            xm = features(bb.generate(a_lo, "demo", args.p, 1000 + i, gm), args.p)
            cols.append((xp - xm) / (2 * args.eps))
        Y = np.stack(cols, axis=1)                 # (d, m)
        G = ridge_operator(Y, V, eps=args.eps)     # (d, T)
        # residual e0: target-feature gap between the swapped anchor and
        # the null-schedule output for the low anchor
        x_null = features(bb.generate(a_lo, "demo", args.p, 1000 + i,
                                      np.ones(args.T)), args.p)
        e0 = np.zeros(G.shape[0])
        e0[:3] = a_hi.mean(axis=(0, 1)) - x_null[:3]
        s = summarize_response(G)
        S = early_alignment(G, e0)
        R_vals.append(s["R"])
        S_vals.append(float(S))
        per_source.append({"source": i, "R": s["R"], "S": float(S)})

    # null-swap floor estimate: spread of per-source R at 5%
    sigma = float(np.std(R_vals)) * 0.05
    res = classify_regime(np.asarray(R_vals), np.asarray(S_vals),
                          sigma_probe=sigma)
    out = {
        "backbone": bb.name,
        "site": bb.condition_site(),
        "sources": per_source,
        "classification": {
            "regime": res.regime.value,
            "R_lo": res.R_lo, "R_hi": res.R_hi,
            "S_lo": res.S_lo, "S_hi": res.S_hi,
            "h": res.h, "reason": res.reason,
        },
        "probe_forward_passes": design.forward_pass_count(),
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out["classification"], indent=2))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
