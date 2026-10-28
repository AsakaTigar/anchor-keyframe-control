# Protocol and arm definitions

## Interfaces (conditioning sites)

| interface | backbone | site | response estimand |
|---|---|---|---|
| Concat | CogVideoX-5B-I2V | interior latent slot (`frame_index` = p) | interior swap response |
| Mask | VACE-1.3B | video + mask at frame p | interior swap response |
| Token | LTX-Video | token interface (`frame_index`) | interior swap response |
| Boundary | CogVideoX / Wan2.2 | official first frame, readout at p | propagated influence (different estimand) |

Interior and boundary rows must not be pooled: the boundary row measures
propagated influence through the rollout, not native interior
reachability.

## Arms

| arm | definition | role |
|---|---|---|
| Native-Full | γ ≡ 1 (official pipeline) | endpoint / fallback |
| Best-Calibration-SkipK | hard skip-K with K from the calibration grid | fixed schedule |
| RAS-OpenLoop-LogSNR | σ(α(logSNR_s − τ)) with frozen (α, τ) | fixed schedule |
| Fixed-Response Gate | one frozen γ vector for every source and seed; no trajectory decode | fixed schedule (not online feedback) |
| ANCHOR (deployed) | regime candidate if the guard certifies; otherwise γ ≡ 1 | controller |
| ANCHOR abstention replay | exact Native-Full records reused when the guard fails | audit row |

## Probe protocol

1. Fix the interface, prompt, seed, and reverse-step count.
2. Draw `m` Rademacher vectors `v_j ∈ {±1}^T` from the recorded seed.
3. Generate `x(γ̄ + ε v_j)` and `x(γ̄ − ε v_j)` for each `j`
   (2m forward passes; no gradients).
4. Reduce each output to the fixed feature map φ; form
   `Y_j = (φ⁺ − φ⁻)/(2ε)`.
5. Recover `Ĝ` (target feature) and `L̂` (off-target feature) by ridge
   regression with the second-difference penalty.
6. Per source compute `R = Σ_s w_s ||Ĝ_s||²`, `U = Σ_s w_s ||L̂_s||²`,
   and `S = ⟨Σ_{s≤K0} Ĝ_s, e0⟩`.
7. Bootstrap over sources; classify with Eq. (3).

## Guard protocol

1. Freeze the candidate schedule and `(α, τ)` before touching target
   calibration data.
2. For each calibration source (target domain), average seeds within the
   source; compute `Δ_i = L_i(candidate) − L_i(native)`.
3. One-sided source-bootstrap upper bound `U_{1−η}` on `E[Δ]`.
4. Deploy the candidate iff `U_{1−η} ≤ δ_NI` (default 0); otherwise
   γ ≡ 1 exactly.

## What is and is not reproduced

- The statistics, classifier, guard, and candidate schedules are fully
  implemented here and unit-tested.
- The per-run CSVs behind Tables II–IV are released with the paper's
  artifact package in the schema used by `scripts/reproduce_table2.py`
  and `scripts/reproduce_table3.py`.
- `examples/*.csv` are synthetic placeholders with the published shape
  and are not measurements.
- The CogVideoX wrapper scales the concatenated condition channels per
  step (`c_s = gamma_s * c(a)`, `c_null = 0`); `gamma = 1` is the native
  pipeline. It does not guarantee bit-exact parity with the authors'
  original harness.
