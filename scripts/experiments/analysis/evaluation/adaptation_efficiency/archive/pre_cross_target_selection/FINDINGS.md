# Adaptation efficiency: pre-cross-target-selection snapshot

This document preserves the original diagnostic summary recovered from the
`prodigy-native-matrix` worktree. It predates the cross-target head-selection
analysis now used as the primary result in
`evaluation/adaptation_efficiency/FINDINGS.md`; the numbers below must not be
treated as the current headline comparison.

Observed 5,472 validation/test rows across 12 model checkpoints and four targets.
All 48 model-target grids were complete.

## Label-efficiency summary

| Family | Mean normalized AUC over log10(labels + 1) | SD | Curves |
|---|---:|---:|---:|
| PRODIGY | 0.7338 | 0.1709 | 36 |
| SAMGPT | 0.6756 | 0.1324 | 36 |
| VISION | 0.6576 | 0.1289 | 36 |
| Raw logistic | 0.6552 | 0.1387 | 12 |
| Raw MLP | 0.6381 | 0.1193 | 12 |
| GraphSAGE | 0.5729 | 0.0594 | 12 |

## Optimization-efficiency diagnostic

Median head updates required to reach 95% of each curve's update-100 ROC-AUC:

| Family | 1 label/class | 10 labels/class | 100 labels/class |
|---|---:|---:|---:|
| PRODIGY | 1.0 | 1.0 | 1.0 |
| VISION | 1.0 | 1.0 | 1.0 |
| SAMGPT | 0.0 | 1.0 | 3.0 |
| GraphSAGE | 0.0 | 0.0 | 1.0 |
| Raw logistic | 1.0 | 1.0 | 1.0 |
| Raw MLP | 0.5 | 1.0 | 1.0 |

All summaries retain every label seed, target, training seed, label budget, and
update cell. The zero-label point is an untrained-head baseline and has no
optimizer updates.

## Provenance

- Recovered from `prodigy-native-matrix/log/adaptation_efficiency/diagnostic_analysis/FINDINGS.md`.
- Preserved during the 2026-09-12 Tucker worktree-storage audit.
- Superseded for primary reporting by the tracked cross-target-selection analysis.
