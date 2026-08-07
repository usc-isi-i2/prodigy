# SAMGPT five-source convergence findings

The five-source Order-C model is already best at the first saved 200-update checkpoint on
the fixed TwiBot-20 validation episodes. ROC-AUC declines slightly as GraphCL loss continues
to improve: `0.7130` at 200 updates versus `0.7098` at 4,000. The matched-wall-clock
2,500-update checkpoint is `0.7104`.

This is negative evidence for using native pretraining loss as a transfer proxy. Between
200 and 4,000 updates the loss falls from `1.26e-3` to `8.00e-7`, while validation AUC falls
by `0.0032`. The best-training-loss model obtains test ROC-AUC `0.7018 ± 0.1212` across 500
episodes.

The run uses one training seed and fixed validation episodes. Episode standard deviation is
not seed-level uncertainty.

Saved evidence:

- `data/validation_trajectory.csv`: all six saved checkpoints.
- `data/metrics.json`: final test result and full protocol.
- `data/provenance.json`: repository and runtime provenance.

Large checkpoints and cached target embeddings remain at the Tucker path recorded in the
paired setup README.
