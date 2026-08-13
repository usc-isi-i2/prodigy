# SAMGPT COVID correctness-ablation findings

## Zero-step baseline

The real graph—not missing-bio leakage—is the main source of the unexpectedly high
checkpoint-zero result.

| representation | validation ROC-AUC |
|---|---:|
| projected features only | 0.5301 |
| random GCN, self-loops only | 0.5160 |
| random GCN, real graph | 0.6718 |
| real graph, nonmissing-bio labeled nodes only | 0.6692 |

Removing missing-bio labeled nodes changes AUC by only `-0.0026`; removing the graph drops
it by `0.1558`.

## Training corrections

| arm | prompt routing | views | loss at 500 | validation ROC-AUC | seconds |
|---|---|---|---:|---:|---:|
| A | inherited behavior | fixed | `3.14e-5` | **0.7010** | 106.9 |
| B | corrected | fixed | `3.65e-9` | 0.6890 | 105.7 |
| C | corrected | resampled each update | `3.11e-9` | 0.6890 | 396.4 |

Correcting structure-prompt routing makes the native objective more than 8,000 times smaller
at 500 updates, but reduces frozen-prototype transfer by `0.0120` AUC. Resampling views is
3.75 times slower than corrected fixed views and changes AUC by only `0.00003`.

Thus fixed augmentations do not explain the fast loss collapse, and lower GraphCL loss does
not imply better frozen downstream transfer. A prompt-aware downstream evaluation remains a
separate open experiment.

Saved evidence:

- `data/zero_step_controls.json`
- `data/arm_b_validation_trajectory.csv`, `data/arm_b_metrics.json`, and provenance
- `data/arm_c_validation_trajectory.csv`, `data/arm_c_metrics.json`, and provenance

All results use one model seed and fixed validation episodes.
