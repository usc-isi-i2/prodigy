# Node Regression benchmark task

Evaluate how well a pretrained model's node representations predict **exogenous
user-profile attributes** — a continuous node-level task that complements the
existing (discrete) classification task.

## Targets

Full profile panel (`--reg-targets`, default = all six):
`followers_count, friends_count, statuses_count, favourites_count, listed_count,
account_age_days`. Values are heavy-tailed → trained on `log1p` (`--reg-transform
log1p`). These are **exogenous** to the retweet graph, so predicting them tests
representation quality rather than echoing adjacency.

## No leakage

Targets live in the graph's `node_targets` field, **outside** `graph.x`. The
encoder only ever sees `--feature_subset emb_only` (bio embeddings), so the target
attribute is never an input feature. Nodes with a missing value (`NaN`) are dropped
from the regression train/val/test splits.

## Metrics

Reported per target: **Spearman ρ** (headline — rank correlation, robust to the
tail), plus RMSE / MAE / R² (see `experiments/trainer.py`).

## Dataset feasibility

Regression runs on **midterm, ukr_rus_twitter, covid19_twitter, twibot20**.
`cp_hk_twitter` has no profile metrics in its source, so it is regression-skipped
automatically (the runner gates on the targets actually present in each artifact).
`twibot20` lacks `favourites_count` (no v2 equivalent) — that target is skipped for it.

## Run

1. **Enrich once** (adds `node_targets`; needs the graph-construction env):

   ```bash
   DATA_ROOT=/dataMeR1/phil/data bash scripts/graph_construction/enrich_all_graphs.sh
   ```

   (Graphs rebuilt from scratch already include the targets by default.)

2. **Evaluate** (prodigy env):

   ```bash
   bash scripts/experiments/node_regression/run_node_regression_eval.sh \
     --checkpoint-run-dir /dataMeR2/phil/gfm/prodigy/state/<run> --gpus 0,1
   ```

   One eval run is produced per (model, dataset, target, shot); the target name is
   encoded in the run-dir prefix (`eval_<model>_to_<ds>_reg_<target>_<shots>shot`).

Results are collected in `scripts/plotting/node_regression/`.
