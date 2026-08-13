# Final-core proper-AUC evidence

This directory is the machine-readable evidence for the proper-AUC extension of
`similarity_vs_transfer_v2`.

- `raw/` preserves the Tucker aggregate, completeness contract, and run
  provenance from `experiment/final-core-auc-grid` at commit `62d9e31`.
- `specialist_cells_three_seed.csv` is the canonical 243-cell table after graph
  names are aligned with the analysis catalog.
- `transfer_matrix_*` files contain three-seed means and sample standard
  deviations for accuracy, macro-F1, and one-vs-rest macro ROC-AUC.
- `predictors/` contains the 9,999-permutation candidate rankings.
- `comparison/` compares the proper final-core AUC ordering with the historical
  9×9 AUC matrix.
- `models/` contains leave-one-graph-out predictions and model rankings.
- `predictability/` tests whether AUC observed on one target forecasts AUC on
  another target.

All headline analyses retain the diagonal in the evidence files but exclude it
from transfer correlations, donor selection, and graph-holdout prediction.
Reproduction commands are in the paired setup README.
