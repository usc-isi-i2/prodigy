# Benchmark-task analysis (node regression + static LP)

`parse_benchmark_eval_logs.py` collects the new benchmark tasks' eval metrics from
Tucker log directories into two tidy CSVs, consumed by the plotting notebooks.

```
python scripts/harness/benchmark_tasks/parse_benchmark_eval_logs.py \
    --log-root /dataMeR2/phil/gfm/prodigy/log \
    --out-dir scripts/experiments/analysis/evaluation/shared_task_tables
```

Outputs:
- `scripts/experiments/analysis/evaluation/shared_task_tables/node_regression/data/node_regression.csv`
  — columns: model, dataset, target, shots, split, spearman, rmse, mae, r2, mse
- `scripts/experiments/analysis/evaluation/shared_task_tables/static_link_prediction/data/static_link_prediction.csv`
  — columns: model, dataset, shots, split, roc_auc, accuracy, f1

It reads the `metrics_<split>[_step<N>].json` files each eval run writes (same
layout as `../export_eval_results_csv.py`), keeping the highest-step metrics per
split, and parses the run-dir name for model / dataset / target / shots.

Then open the notebooks in `scripts/experiments/analysis/evaluation/shared_task_tables/{node_regression,static_link_prediction}/`.
