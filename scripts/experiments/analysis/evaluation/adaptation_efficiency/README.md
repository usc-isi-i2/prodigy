# Frozen-encoder adaptation-efficiency analysis

The executable protocol lives in
[`../../../setup/adaptation_efficiency/`](../../../setup/adaptation_efficiency/).
After Tucker exports `results.csv`, run `analyze_results.py` to preserve the full
budget-by-update learning curves and produce:

- ROC-AUC, accuracy, and macro-F1 summaries;
- label-efficiency area under ROC-AUC versus log label budget;
- first head update reaching 95% of update-100 performance;
- coverage checks for every valid budget-by-update-by-seed cell;
- PNG and PDF figures plus machine-readable CSV/JSON summaries.

Zero-label rows are valid only at update 0. The analysis rejects the idea of a
zero-label optimizer trajectory and does not reduce the result to each model's
best checkpoint.
