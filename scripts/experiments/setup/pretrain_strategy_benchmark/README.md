# Pretraining-strategy benchmark

Compare **pretraining strategies** — models trained on the same covid retweet
data with different self-supervised objectives — on the retweet-net benchmark, to
see which pretraining task yields the most transferable representations.

## Strategies (existing checkpoints)

Same data (covid), different pretraining objective, `emb_dim=256`, step 11000
(`model_list.txt`):

| id | pretraining objective |
|----|-----------------------|
| `task_transfer_covid_nm` | neighbor matching |
| `task_transfer_covid_cl` | contrastive |
| `task_transfer_covid_fp` | (masked) feature prediction |

Swap the model list to compare other strategies (e.g. the `nm_cm_*` source-mix runs).

## Tasks evaluated

All frozen-encoder, few/zero-shot eval (`--eval_only`), across all 5 datasets:

- **node regression** (headline) — profile panel, log1p, Spearman; 10-shot.
- **static link prediction** (headline) — held-out edges vs 2-hop hard negatives,
  ROC-AUC; **zero-shot + `--slp-n-query 4`** (sparse-graph safe).
- **neighbor matching** + **classification** — baselines, 3-shot (nm is near-ceiling;
  classification only where labels exist, i.e. twibot20).

The two new tasks give the cleanest strategy signal (nm saturates; pl is single-dataset).

## Run

```bash
# once: enrich graphs with benchmark targets
DATA_ROOT=/dataMeR1/phil/data bash scripts/graph_construction/enrich_all_graphs.sh
# then: the strategy sweep (prodigy env; ~GPUs 0-3)
bash scripts/experiments/setup/pretrain_strategy_benchmark/run_pretrain_strategy_benchmark.sh --gpus 0,1,2,3
```

Results (keyed by `model` = strategy) land in
`scripts/experiments/analysis/evaluation/task_tables/{node_regression,static_link_prediction}/data/*.csv`; the
strategy comparison notebook is in
`scripts/experiments/analysis/objectives/legacy/pretrain_strategy_benchmark/`.
