# ANCHOR hyperparameters

Two categories are distinguished: **paper-fixed** values reported in the
manuscript, and **released defaults** chosen for this reference
implementation where the manuscript defers to the supplement or does
not specify a number. Released defaults are parameters of *this*
implementation, not claims about the authors' original runs.

## Generation protocol

| item | value | source |
|---|---|---|
| backbone (main) | CogVideoX-5B-I2V | paper §V |
| resolution / frames | 480 × 720 × 49 | paper §V |
| sampler | DDIM | paper §V |
| reverse steps T | 30 | paper §V |
| guidance scale | 6.0 | released default |
| requested anchor frame p | 24 | paper §V |
| seeds | 42, 123 (two seeds per source) | paper §V |

## Probe (Stage 1)

| item | value | source |
|---|---|---|
| design | Rademacher v ∈ {−1,+1}^T, symmetric ±ε | paper Eq. (2) |
| ε | 0.1 | released default (probe radius in schedule space) |
| ridge μ on second difference | 1e-2 | released default |
| probe pairs m | 24 (budget sweep: 8…24; recovery poor at 8) | paper §III-A |
| early window K0 | ⌈T/6⌉ = 5 | paper §III-A |
| forward passes per probe | 2 (paired; no back-prop) | paper Fig. 1 |
| probe budget total | 2m = 48 generations | derived |

## Classification (Stage 2)

| item | value | source |
|---|---|---|
| noise floor | h = 3·σ̂²_probe | paper §III-B |
| confidence level | 95 % two-sided for R and S | paper Eq. (3) |
| bootstrap | BCa over sources; B = 10⁴; seed 2027 | paper §III-B |
| minimum sources | 3 | released default |

## Guard (Stage 3)

| item | value | source |
|---|---|---|
| margin δ_NI | 0 (conservative) | paper Eq. (6) |
| bound | one-sided source-bootstrap upper bound, level 1−η | paper Eq. (6) |
| η | 0.05 | paper Table III |
| seeds per source | averaged before inference | paper §V |
| missing / failed certificate | γ = 1 (Native-Full) | paper Eq. (6) |

## Candidates (Eq. 4)

| regime | schedule | notes |
|---|---|---|
| Wall / Uncertain | γ ≡ 1 | Native-Full |
| Persistent | 1[s > 1] (micro-skip K = 1) | paper §III-C |
| Droppable | σ(α(logSNR_s − τ)) | (α, τ) frozen before target evaluation |
| locked-lane candidate | skip-K = 8 | paper Table III |
| τ | −4.495 | paper §V (frozen from T = 30 calibration) |
| α | 4.0 | released default |

## Statistics in the tables

| item | value | source |
|---|---|---|
| SSv2 panel | 27 sources × 2 seeds × 4 arms | paper Table II |
| DAVIS lane | 10 train + 10 val sources × 2 seeds | paper Table III |
| interval | BCa 95 %, source-level | paper Tables II–III |
| W/T/L | per source, candidate vs Native-Full | paper Tables II–III |
| multiplicity | Holm across the three SSv2 fixed arms | paper Table II |
| sign-flip permutation | B = 2·10⁴ | released default |
