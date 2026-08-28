# Adaptation-efficiency results

Observed 5,472 validation/test rows across 12 model checkpoints and 4 targets. Complete model-target grids: 48/48.

## Primary cross-target-selected label-efficiency summary

The zero-label baseline plus the leave-one-target-graph-out selected positive-label points define this label-efficiency summary. The fixed-update-100 curve is retained separately as a legacy diagnostic.

| Family | mean normalized AUC over log10(labels + 1) | SD | curves |
|---|---:|---:|---:|
| PRODIGY | 0.7475 | 0.1749 | 36 |
| VISION | 0.6658 | 0.1326 | 36 |
| SAMGPT | 0.6610 | 0.1271 | 36 |
| Raw MLP | 0.6609 | 0.1419 | 12 |
| Raw logistic | 0.6540 | 0.1378 | 12 |
| GraphSAGE | 0.5704 | 0.0502 | 12 |

## Primary selection protocol

The primary few-shot result uses leave-one-target-graph-out development selection: for each target and label budget, one update count is selected from family-balanced validation performance on the other three targets and then shared by every model family. The target's own validation labels and all test labels are excluded from selection. Target-validation selection is retained only as an oracle diagnostic.

| Family | 1 label/class | 10 labels/class | 100 labels/class |
|---|---:|---:|---:|
| PRODIGY | 0.7210 | 0.7741 | 0.8050 |
| VISION | 0.6306 | 0.6830 | 0.7336 |
| SAMGPT | 0.5582 | 0.6808 | 0.7701 |
| GraphSAGE | 0.5432 | 0.5733 | 0.6106 |
| Raw logistic | 0.6204 | 0.6749 | 0.7139 |
| Raw MLP | 0.6316 | 0.6713 | 0.7120 |

For future unseen targets, the locked shared schedule is 1 label/class → 3 updates, 10 labels/class → 10 updates, and 100 labels/class → 10 updates.

## Late-training diagnostic

From update 10 to 100, labeled-training loss fell in 16/18 family-by-budget curves while test ROC-AUC fell in 14/18. Falling training loss alongside falling validation/test performance is evidence of head overfitting, not optimizer divergence or encoder drift.

| Family | labels/class | Δ train loss (100−10) | Δ test AUC (100−10) | validation-selection gain vs 100 |
|---|---:|---:|---:|---:|
| GraphSAGE | 1 | -0.0775 | -0.0012 | +0.0337 |
| GraphSAGE | 10 | -0.0857 | -0.0079 | +0.0405 |
| GraphSAGE | 100 | -0.0241 | +0.0193 | +0.0039 |
| PRODIGY | 1 | -0.0017 | -0.0001 | +0.0148 |
| PRODIGY | 10 | -0.1053 | -0.0126 | +0.0268 |
| PRODIGY | 100 | -0.1631 | -0.0215 | +0.0226 |
| Raw MLP | 1 | +0.0000 | -0.0250 | +0.0567 |
| Raw MLP | 10 | -0.0221 | -0.0117 | +0.0364 |
| Raw MLP | 100 | -0.2382 | -0.0109 | +0.0271 |
| Raw logistic | 1 | +0.0000 | +0.0016 | +0.0044 |
| Raw logistic | 10 | -0.0014 | -0.0005 | +0.0023 |
| Raw logistic | 100 | -0.0694 | -0.0027 | +0.0179 |
| SAMGPT | 1 | -0.0068 | +0.0005 | +0.0611 |
| SAMGPT | 10 | -0.2180 | +0.0011 | +0.0265 |
| SAMGPT | 100 | -0.1948 | -0.0054 | +0.0150 |
| VISION | 1 | -0.0000 | -0.0000 | +0.0069 |
| VISION | 10 | -0.0129 | -0.0038 | +0.0148 |
| VISION | 100 | -0.2431 | -0.0218 | +0.0290 |

## Optimization-efficiency summary

Median head updates required to reach 95% of each curve's update-100 ROC-AUC:

| Family | 1 label/class | 10 labels/class | 100 labels/class |
|---|---:|---:|---:|
| PRODIGY | 1.0 | 1.0 | 1.0 |
| VISION | 1.0 | 1.0 | 1.0 |
| SAMGPT | 0.0 | 1.0 | 3.0 |
| GraphSAGE | 0.0 | 0.0 | 1.0 |
| Raw logistic | 1.0 | 1.0 | 1.0 |
| Raw MLP | 0.5 | 1.0 | 1.0 |

All summaries retain every label-seed, target, training-seed, label-budget, and update cell. The zero-label point is an untrained-head baseline and has no optimizer updates.
