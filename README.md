# ANCHOR — reference implementation

Released with the IEEE Transactions on Consumer Electronics revision of

> **ANCHOR: Probe, Classify, Gate — Interface- and Domain-Adaptive
> Keyframe Control for Video Diffusion**

Repository: <https://github.com/AsakaTigar/anchor-keyframe-control>

This repository contains the controller and its statistics, written to
be re-implementable from the paper alone:

| component | module | paper reference |
|---|---|---|
| signed paired probe (Rademacher, ridge + 2nd-difference) | `anchor/probe.py` | Stage 1, Eq. (2) |
| source-level BCa bootstrap | `anchor/bootstrap.py` | §III-B, Stage 2/3 |
| four-way classifier (Wall / Persistent / Droppable / Uncertain) | `anchor/classify.py` | Eq. (3) |
| target-domain non-inferiority guard | `anchor/guard.py` | Eq. (6) |
| candidate schedules (native, skip-K, log-SNR sigmoid) | `anchor/candidates.py` | Eq. (4), Prop. 3 |
| metrics (anchor LPIPS, temporal consistency, leakage) | `anchor/metrics.py` | §V |
| backbone adapters (no weights shipped) | `anchor/backbones/` | §III-A |

**No third-party video-diffusion weights are redistributed.** Backbones
are accessed through the adapter protocol
(`anchor/backbones/base.py`); a CogVideoX-5B-I2V wrapper is provided
and expects a local model copy (`COGVIDEOX_I2V_DIR`).

## Install

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt          # numpy scipy (CPU: probe/statistics)
# optional, for LPIPS + backbone wrappers:
pip install -r requirements-gpu.txt      # torch diffusers lpips opencv-python
```

## Quickstart (no GPU, ~10 s)

```bash
# 1) probe -> classify on the synthetic adapter
python scripts/run_probe.py --backbone synthetic --sources 6 --m 16 --T 12

# 2) guard on a calibration CSV
python scripts/run_guard.py --csv examples/davis_lane_example.csv \
    --candidate candidate --native native --delta-NI 0

# 3) reproduce the DAVIS 10+10 table from a per-run CSV
python scripts/reproduce_table3.py --csv examples/davis_lane_example.csv

# 4) reproduce the SSv2 source-level table from a per-run CSV
python scripts/reproduce_table2.py --csv examples/ssv2_example.csv

# 5) unit tests
python -m pytest -q tests/
```

`examples/*.csv` are **synthetic values with the published shape**
(10+10 DAVIS sources, 27 SSv2 sources × 2 seeds × 4 arms) so that every
script runs end-to-end without restricted data. They are clearly not
the measured numbers; the per-run manifests that produced Tables II–IV
accompany the paper's artifact package in the same CSV schema.

## Real backbones

```bash
export COGVIDEOX_I2V_DIR=/path/to/CogVideoX-5b-I2V
python scripts/run_probe.py --backbone cogvideox --sources 5 --m 24 \
    --anchors /path/to/davis_first_frames --p 24 --T 30
```

The CogVideoX wrapper scales the native condition channels per denoising
step: CogVideoX I2V concatenates the encoded anchor latents to the noisy
latents along the channel axis, and the wrapper multiplies that
condition half by `gamma_s` for the current step. `gamma == 1` leaves the
native pipeline bit-identical; `gamma == 0` removes the condition
(zero-latent null path). This is a documented reimplementation of the
interpolant in Eq. (1); bit-exact parity with the authors' original hook
is not claimed — the released per-step γ vectors and run manifests are
the reference for the reported numbers.

## Repo layout

```
anchor/                 controller + statistics (importable package)
  backbones/            adapter protocol, synthetic + CogVideoX wrappers
scripts/                run_probe, run_guard, reproduce_table2/3
examples/              synthetic CSVs with the published schema
tests/                 pytest suite (probe recovery, classifier, guard, BCa)
docs/PARAMETERS.md     every hyperparameter, paper-fixed vs released default
docs/PROTOCOL.md       generation protocol and arm definitions
```

## What is verified here

- `tests/test_core.py` checks that the ridge probe recovers a known
  smooth operator, that the classifier implements all four quadrants,
  that the guard deploys/abstains correctly, and that BCa intervals
  behave on synthetic samples.
- The table scripts are exercised on the synthetic example CSVs.

## License

Code: MIT. The paper text and figures are not covered by this license.
Third-party datasets (DAVIS, SSv2) keep their own licenses; no video
frames are redistributed here.
