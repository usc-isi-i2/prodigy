# Findings: one-seed, 100-update architecture matrix

## Registered scope

- Architectures: PRODIGY, VISION, GILT.
- Source sets: the same 31 final-core physical sets.
- Training: seed 0, 100 optimizer updates, effective batch size 4 (400 neighbor-matching episodes).
- Evaluation: 128 fixed 2-way/10-shot binary-classification episodes on four targets; ROC-AUC primary.
- Completeness: 372/372 architecture × source-set × target cells.
- Episode audit: one support/query-node fingerprint per target across all architectures and models.

This is a descriptive low-budget system comparison, not a converged architecture ranking or a parameter-matched backbone ablation.

## Aggregate result

Mean ROC-AUC over all 124 model-target cells per architecture:

| Architecture | Mean ROC-AUC | Specialist mean | Mixture mean |
|---|---:|---:|---:|
| VISION | 0.7391 | 0.7118 | 0.7502 |
| GILT | 0.6831 | 0.6511 | 0.6963 |
| PRODIGY | 0.5840 | 0.5963 | 0.5790 |

Paired over the same 124 model-target cells, VISION minus PRODIGY averages +0.1550 ROC-AUC (positive in 71.0% of cells), GILT minus PRODIGY +0.0991 (58.1%), and GILT minus VISION -0.0559 (GILT positive in 8.9%). These cells are dependent measurements from one training seed and four fixed evaluation streams, not 124 independent replicates.

The architecture effect is target-dependent. VISION/GILT are strongest on Covid Political and Election 2020, whereas PRODIGY is strongest on TwiBot-20; all systems are near chance on Ukraine Suspended at this budget.

## Composition rule

The best-included-specialist envelope recurs strongly for VISION and GILT but not for undertrained PRODIGY:

| Architecture | Best-specialist MAE | Pooled r | Target-demeaned r | Mean residual |
|---|---:|---:|---:|---:|
| VISION | 0.0192 | 0.997 | 0.623 | -0.0186 |
| GILT | 0.0304 | 0.993 | 0.866 | -0.0304 |
| PRODIGY | 0.1471 | 0.243 | -0.094 | -0.1386 |

Thus the primary 2,500-update PRODIGY composition result is not an architecture-free property at update 100: it appears in VISION/GILT immediately, while PRODIGY has not yet learned a stable source-composition geometry.

## Budget sensitivity

The preserved 500-update single-source (`ss_ukr_rus`) pilot uses the same four targets and evaluation fingerprints. Its architecture means are PRODIGY 0.7592, VISION 0.7578, and GILT 0.6829, versus VISION 0.6609, GILT 0.6261, and PRODIGY 0.5896 at the 100-update pilot. The rank change on one source set confirms that 100 updates is pre-saturation and prevents a claim of architecture superiority.

## Protocol boundaries

- Parameter counts differ: PRODIGY 1.64M, VISION 2.05M, GILT 2.12M.
- PRODIGY uses directed sampled edges and raw 768D features; VISION/GILT symmetrize, VISION centers 768D features, and GILT uses one frozen seed-0 orthogonal 768→128 projection.
- Label mechanisms, losses, and optimizers remain architecture-native.
- Memory-bound adapter runs accumulate four per-episode gradients before one optimizer step, preserving the effective batch and update count.
- One seed and unequal target query counts rule out confidence intervals or a converged leaderboard interpretation.
