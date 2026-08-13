# Log-recovered PRODIGY metrics

`physical_metrics.tsv` preserves the accuracy, macro-F1, and multiclass
one-vs-rest macro ROC-AUC lines emitted by the original fixed-test workers.
Together, the production, recovery, and continuation logs contain exactly one
completed metric record for each of the 837 physical PRODIGY evaluation cells.

The trainer printed these values to four decimal places. The table is therefore
appropriate for visualization and ordinary summary analysis but is not a
full-precision replacement for the original metric sidecars. Its logged
specialist AUC values agree with all 243 full-precision specialist replay JSONs
to the expected rounding tolerance (at most 0.00005).

Regenerate the table read-only from Tucker with the script's default SSH mode:

```bash
/opt/homebrew/bin/python3.11 \
  scripts/experiments/analysis/transfer/matrices/native_objective/final_core/recover_logged_prodigy_metrics.py
```

The source log for every row is recorded as an absolute Tucker path. No
training or evaluation is rerun by this recovery step.
