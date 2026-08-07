# SAMGPT sampled-COVID saturation findings

Transfer improves quickly and is effectively saturated by 500–1,000 updates.

| updates | validation ROC-AUC | elapsed seconds |
|---:|---:|---:|
| 0 | 0.67179 | 0.0 |
| 100 | 0.69754 | 21.7 |
| 500 | 0.70100 | 106.9 |
| 1,000 | 0.70201 | 213.2 |
| 2,000 | 0.70243 | 425.9 |
| 4,000 | 0.70253 | 851.1 |

The 2,000→4,000 gain is only `0.00010` AUC even though the native loss falls another
4.7-fold. The best-training-loss checkpoint obtains held-out test ROC-AUC
`0.68756 ± 0.12516` across 500 episodes. For this sampled graph, 500 or 1,000 updates is a
more defensible standard budget than 4,000.

The checkpoint-zero result is not a chance classifier: it includes node features, the real
target topology, a random GCN, and labeled support prototypes. The paired correctness
ablation isolates those contributions.

Saved evidence:

- `data/validation_trajectory.csv`
- `data/metrics.json`
- `data/provenance.json`

The run uses one model seed and fixed episodes; reported episode variation is not seed-level
uncertainty.
