# SAMGPT native-GraphCL ladder findings

This paired folder archives the native-objective SAMGPT ladder result separately from the
PRODIGY order-robustness analysis where it was first assembled.

Across 27 ladder models × 9 targets, adding a target graph to the training mixture lowers
its own GraphCL BCE in 21 of 24 eligible graph-entry comparisons. The specialist-maximum
rule is much closer to the observed ladder behavior than the specialist mean, while the
probability-margin view avoids the severe accuracy ceiling of the easy corruption task.

The result is descriptive evidence from one seed and one fixed unseen GraphCL view per
target. It should not be reported as seed-level uncertainty, and raw classification
accuracy is a poor primary measure because many cells are at or near 1.0.

Saved evidence:

- `data/cells.csv`: long-form ladder cells used in the max-rule comparison.
- `data/summary.csv`: per-order/rung/target rule comparison.
- `data/rule_comparison_summary.csv`: aggregate max-versus-mean errors.
- `data/manifest.json`: source exports and provenance.
- `data/graphcl_source_losses.csv` and `data/graphcl_loss_summary.csv`: native losses.

The original extended write-up remains in
`analysis/nm_ladder_order_robustness/FINDINGS_GRAPHCL_NATIVE.md`; this folder is the
name-aligned archival record.
