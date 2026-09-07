# Frozen-encoder adaptation-efficiency analysis

The executable protocol lives in
[`../../../setup/adaptation_efficiency/`](../../../setup/adaptation_efficiency/).
After Tucker exports `adaptation_cells.csv`, run `analyze_results.py` to preserve the full
budget-by-update learning curves and produce:

- ROC-AUC, accuracy, and macro-F1 summaries;
- label-efficiency area under ROC-AUC versus log label budget;
- first head update reaching 95% of update-100 performance;
- labeled-train loss and ROC-AUC trajectories for diagnosing late degradation;
- validation-selected updates with untouched test performance;
- leave-one-target-graph-out update selection shared across model families (primary);
- coverage checks for every valid budget-by-update-by-seed cell;
- PNG and PDF figures plus machine-readable CSV/JSON summaries.

`plot_selection_protocol.py` produces the paper-facing two-panel comparison of the
primary cross-target-selected result and its change relative to the legacy fixed-100
endpoint (`figures/selection_protocol_comparison.{png,pdf}`).

The analyzer requires exactly 5,472 validation/test rows: 12 registered model
IDs × four targets × three label seeds × 19 valid budget/update cells × two
splits. It fails on missing or duplicate cells and copies the full unaggregated
grid into the analysis `data/` folder before producing family summaries.

Zero-label rows are valid only at update 0. The analysis rejects the idea of a
zero-label optimizer trajectory and does not reduce the result to each model's
best checkpoint.

For the primary few-shot comparison, the analyzer excludes the reported target's
validation split when choosing the head-update milestone. It averages validation
performance over the other three targets with equal model-family and target weight,
chooses the earliest update on a tie, and applies that one milestone to every family on
the held-out target. Per-target validation selection is an explicitly secondary oracle.
