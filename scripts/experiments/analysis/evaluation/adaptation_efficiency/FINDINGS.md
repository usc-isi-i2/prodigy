# Adaptation-efficiency results

Observed 3,744 validation/test rows across 12 model checkpoints and 4 targets. Complete model-target grids: 48/48.

## Label-efficiency summary

| Family | mean normalized AUC over log10(labels + 1) | SD | curves |
|---|---:|---:|---:|
| PRODIGY | 0.7338 | 0.1709 | 36 |
| SAMGPT | 0.6756 | 0.1324 | 36 |
| VISION | 0.6576 | 0.1289 | 36 |
| Raw logistic | 0.6552 | 0.1387 | 12 |
| Raw MLP | 0.6381 | 0.1193 | 12 |
| GraphSAGE | 0.5729 | 0.0594 | 12 |

## Optimization-efficiency summary

Median head updates required to reach 95% of each curve's update-100 ROC-AUC:

| Family | 1 label/class | 10 labels/class | 100 labels/class |
|---|---:|---:|---:|
| PRODIGY | 1.0 | 1.0 | 1.0 |
| VISION | 1.0 | 1.0 | 1.0 |
| SAMGPT | 0.0 | 1.0 | 10.0 |
| GraphSAGE | 0.0 | 0.0 | 1.0 |
| Raw logistic | 1.0 | 1.0 | 1.0 |
| Raw MLP | 0.5 | 1.0 | 1.0 |

All summaries retain every label-seed, target, training-seed, label-budget, and update cell. The zero-label point is an untrained-head baseline and has no optimizer updates.
