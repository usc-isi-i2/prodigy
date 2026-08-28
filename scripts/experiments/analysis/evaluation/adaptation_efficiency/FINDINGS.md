# Adaptation-efficiency results

Observed 5,472 validation/test rows across 12 model checkpoints and 4 targets. Complete model-target grids: 48/48.

## Label-efficiency summary

| Family | mean normalized AUC over log10(labels + 1) | SD | curves |
|---|---:|---:|---:|
| PRODIGY | 0.7338 | 0.1709 | 36 |
| SAMGPT | 0.6756 | 0.1324 | 36 |
| VISION | 0.6576 | 0.1289 | 36 |
| Raw logistic | 0.6552 | 0.1387 | 12 |
| Raw MLP | 0.6381 | 0.1193 | 12 |
| GraphSAGE | 0.5729 | 0.0594 | 12 |

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
