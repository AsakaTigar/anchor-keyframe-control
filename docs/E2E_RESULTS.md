# End-to-end pipeline test — run record (2026-10-29)

Environment: HZU a100_2, GPU1 (A100-80GB), torch 2.14.1+cu130,
diffusers 0.38.0, transformers 4.57.1. Weights: ModelScope
`AI-ModelScope/CogVideoX-5b-I2V`; DINOv2 `facebook/dinov2-base`.

Protocol: 3 calibration + 3 held-out DAVIS first-frame sources, one
prompt, seed 42, T=30, p=24, 480x720x49, m=4 probe pairs, LPIPS-Alex.

| stage | outcome |
|---|---|
| probe | Persistent; S_lo=0.262, R_lo=0.211 (per-source S +0.26/+1.18/+1.40) |
| candidate | micro-skip K=1 (proposed by the regime) |
| guard (delta_NI=0) | deltas +0.034/-0.036/-0.056; U=+0.011 -> abstain |
| deployed | Native-Full (gamma = 1) |
| post-veto held-out | dLPIPS +0.096/+0.377/+0.049 (3/3 worse), mean +0.174; dDINO -0.431 |
| cost | 30 generations, 5,521 s, 164.4+/-1.0 s/gen, 37.5 GB peak |

The pipeline executed probe -> classify -> propose -> certify and the
guard vetoed. The post-veto diagnostic shows the veto was protective.
No adaptive win is claimed; a positive case remains open.

Environment note: the same probe on the earlier GDUT a100 stack gave
(R,S)=(0.453,+0.09) for source 1 and (2.496,-1.39) for source 2 vs
(0.211,+0.26) and (1.312,+1.18) here — source 2's sign flips, so
certificates are bound to the environment/stack as well as the model.
