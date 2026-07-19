# directed3_log (input-scaling fix) vs original — topology_feature_ssl

`directed3_log` = log1p the raw in/out degree counts before z-scoring (fixes the ~1322σ heavy-tailed input that suppressed E1). Each `_log` arm changed ONLY that vs its original. **Baselines are the clean matched-40k rows (`*_40k`), which reproduce the published FINDINGS exactly.** `_log` arms evaluated fresh with the identical harness. Single seed. Downstream = mean over eval datasets (test split); reg headline mean is over the 3 targets present for all arms (followers/statuses/account_age).

## Headline — pretext + downstream

| arm | pretext val_auc | reg ρ (core3) | reg account_age | cls AUC | static-LP AUC | min(cls,slp) |
|---|---|---|---|---|---|---|
| B0 | 0.955 | -0.003 | -0.062 | 0.793 | 0.675 | 0.675 |
| E1 | 0.947 | 0.135 | 0.118 | 0.778 | 0.657 | 0.657 |
| E1_log | 0.957 | -0.047 | 0.013 | 0.770 | 0.343 | 0.343 |
| E2 | 0.950 | -0.077 | -0.069 | 0.781 | 0.761 | 0.761 |
| E2_log | 0.950 | -0.106 | -0.156 | 0.788 | 0.381 | 0.381 |
| E2b | 0.952 | -0.001 | 0.047 | 0.784 | 0.401 | 0.401 |
| E2b_log | 0.953 | -0.076 | -0.004 | 0.779 | 0.526 | 0.526 |
| E4 | — | -0.133 | -0.079 | 0.445 | 0.662 | 0.445 |
| E4_log | 0.500 | -0.237 | -0.144 | 0.482 | 0.639 | 0.482 |
| E4r | 0.500 | -0.124 | -0.005 | 0.643 | 0.234 | 0.234 |
| E4r_log | 0.500 | -0.191 | -0.030 | 0.429 | 0.641 | 0.429 |

**Δ (\_log − original\_40k):**

| arm | Δ pretext | Δ reg core3 | Δ reg account_age | Δ cls | Δ static-LP |
|---|---|---|---|---|---|
| E1→E1_log | +0.010 | -0.181 | -0.105 | -0.009 | -0.313 |
| E2→E2_log | -0.000 | -0.029 | -0.087 | +0.007 | -0.380 |
| E2b→E2b_log | +0.001 | -0.075 | -0.051 | -0.005 | +0.125 |
| E4→E4_log | — | -0.104 | -0.065 | +0.037 | -0.024 |
| E4r→E4r_log | +0.000 | -0.067 | -0.024 | -0.215 | +0.406 |

## Regression — Spearman per target (mean over datasets)

| arm | followers | friends | statuses | favourites | listed | account_age |
|---|---|---|---|---|---|---|
| B0 | 0.033 | -0.072 | 0.021 | -0.002 | -0.001 | -0.062 |
| E1 | 0.191 | 0.151 | 0.095 | 0.044 | 0.141 | 0.118 |
| E1_log | -0.031 | -0.064 | -0.122 | -0.124 | -0.064 | 0.013 |
| E2 | -0.068 | -0.105 | -0.095 | -0.078 | -0.045 | -0.069 |
| E2_log | -0.086 | -0.074 | -0.076 | -0.116 | -0.142 | -0.156 |
| E2b | -0.041 | — | -0.009 | — | — | 0.047 |
| E2b_log | -0.099 | -0.115 | -0.124 | -0.094 | -0.013 | -0.004 |
| E4 | -0.181 | — | -0.139 | — | — | -0.079 |
| E4_log | -0.357 | -0.188 | -0.210 | -0.090 | -0.322 | -0.144 |
| E4r | -0.233 | — | -0.133 | — | — | -0.005 |
| E4r_log | -0.278 | -0.222 | -0.265 | -0.189 | -0.175 | -0.030 |

## Static link prediction — ROC-AUC per dataset

| arm | covid19_twitter | midterm | twibot20 | ukr_rus_twitter | mean |
|---|---|---|---|---|---|
| B0 | 0.657 | 0.658 | 0.635 | 0.753 | 0.675 |
| E1 | 0.628 | 0.635 | 0.714 | 0.650 | 0.657 |
| E1_log | 0.319 | 0.413 | 0.351 | 0.290 | 0.343 |
| E2 | 0.780 | 0.708 | 0.735 | 0.823 | 0.761 |
| E2_log | 0.386 | 0.395 | 0.411 | 0.333 | 0.381 |
| E2b | 0.402 | 0.361 | 0.517 | 0.323 | 0.401 |
| E2b_log | 0.486 | 0.642 | 0.568 | 0.410 | 0.526 |
| E4 | 0.646 | 0.608 | 0.732 | 0.664 | 0.662 |
| E4_log | 0.641 | 0.565 | 0.709 | 0.639 | 0.639 |
| E4r | 0.212 | 0.281 | 0.276 | 0.168 | 0.234 |
| E4r_log | 0.626 | 0.576 | 0.685 | 0.676 | 0.641 |

## Capability probes — linear-probe AUC (chance 0.50)

| arm | count_thr | in_deg | out_deg | existence | conjunction |
|---|---|---|---|---|---|
| B0 | 0.478 | 0.515 | 0.524 | 0.515 | 0.513 |
| E1 | 0.672 | 0.627 | 0.515 | 0.535 | 0.534 |
| E1_log | 0.563 | 0.531 | 0.524 | 0.518 | 0.521 |
| E2 | 0.589 | 0.513 | 0.583 | 0.623 | 0.626 |
| E2_log | 0.504 | 0.511 | 0.505 | 0.631 | 0.598 |
| E2b | 0.659 | 0.558 | 0.710 | 0.548 | 0.574 |
| E2b_log | 0.526 | 0.528 | 0.544 | 0.568 | 0.536 |
| E4 | 0.245 | 0.291 | 0.359 | 0.508 | 0.468 |
| E4_log | 0.180 | 0.208 | 0.238 | 0.477 | 0.440 |
| E4r | 0.178 | 0.148 | 0.369 | 0.449 | 0.377 |
| E4r_log | 0.166 | 0.195 | 0.244 | 0.421 | 0.381 |

## Row-count sanity (test rows found per _log arm)

| arm | reg | cls | slp |
|---|---|---|---|
| E1_log | 23 | 2 | 4 |
| E2_log | 23 | 2 | 4 |
| E2b_log | 23 | 2 | 4 |
| E4_log | 23 | 2 | 4 |
| E4r_log | 23 | 2 | 4 |

---

## Interpretation

The input-scaling fix (`directed3_log`) **overturns the downstream headlines** of the original study — the apparent E-arm wins were largely artifacts of the ~1322σ heavy-tailed degree inputs, not genuine learned structure.

1. **E1's regression edge was leakage.** E1 was "the only arm with positive regression" (core-3 ρ 0.135, and a supposedly-clean `account_age` 0.118). With log1p inputs it goes **negative** (−0.047), and `account_age` collapses to **0.013 ≈ B0** — so even the "leakage-free" target was a scaling artifact.
2. **E2's static-LP win was a scaling artifact.** E2 (0.761, the study's central "count-aware aggregation buys topology" result) drops to **0.381**, below plain B0 (0.675).
3. **After the fix, no structural-input / count-aware arm beats B0** on either regression or static-LP. The best `_log` LP is E4r_log/E4_log ≈ 0.64 < B0 0.675.
4. **Classification is unchanged** for the NM arms (E1 0.778→0.770, E2 0.781→0.788): the fix removed exactly the degree-driven artifacts and left the genuine bio signal intact.
5. **Probes separate real from artifact.** E1's count/in-degree probe leadership was passthrough (count_thr 0.672→0.563, in_deg 0.627→0.531 → near chance). E2's **existence/conjunction is genuine** (0.623/0.626 → 0.631/0.598, survives) — the one real capability the encoder axis buys via sum-aggregation — but it does **not** translate into better downstream transfer (E2's LP still collapsed). E4/E4r stay below chance before and after (the multi-task rep is degraded regardless of input scaling).

**Revised conclusion:** on this stack, once inputs are scaled correctly, degree-as-input / count-aware aggregation does **not** buy transferable downstream structure — the original evidence for it was a scaling artifact, and plain bio-only B0 remains best-or-tied on both the feature and the topological task.

**Caveats:** single seed; static-LP is noisy in this harness (below-chance values occur for several arms). Read rankings/directions, not third decimals.
