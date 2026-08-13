# NM single-source downstream matrix

## Question

Do the eight matched-40k single-source neighbor-matching (NM) encoders differ in
their transfer to downstream node classification and node regression?

This complements `analysis/transfer/matrices/prodigy_nm/single_source/nm_single_source_matrix`, which evaluates the same
source models on the NM pretraining task. The outputs here are downstream-task
matrices:

- node classification: 8 source models × 4 labeled graphs, 10-shot ROC-AUC;
- node regression: 8 source models × 4 profile graphs × 3 targets, 10-shot
  Spearman with `log1p` targets.

## Checkpoint policy

All scores use the eight original `state_dict_40000.ckpt` files that produced
the NM single-source transfer matrix. They live in the main Tucker checkout's
`state/` directory under `nm_ss_<dataset>_<timestamp>/checkpoint/`; this
experiment resolves those exact timestamped run directories and records their
absolute paths in `data/model_manifest.csv`.

No training is required. In particular, the Ukraine row uses
`nm_ss_ukr_rus_twitter`, not the older `ukr_only_nm` checkpoint used as rung 1
in some ladder analyses.

## Eligible evaluation graphs

The graph catalog determines which downstream tasks are valid:

- regression: `ukr_rus_twitter`, `covid19_twitter`, `midterm`, `twibot20`;
  targets `followers_count`, `statuses_count`, `account_age_days`;
- classification: `covid_political`, `election2020`,
  `ukr_rus_suspended`, `twibot20`.

Eval uses the shared frozen-encoder runner and the same 10-shot/log1p protocol
as `nm_ladder_downstream`.

## Run on Tucker

Use an isolated worktree. From the worktree:

```bash
tmux new-session -d -s nmssd_pipeline \
  'export PATH="/home/mhchu/miniconda3/bin:$PATH"; \
   EVAL_GPUS="0,1,2,3" \
   bash scripts/experiments/setup/nm_single_source_downstream/run_pipeline_tucker.sh'
```

Progress:

```bash
cat scripts/experiments/setup/nm_single_source_downstream/run_logs/pipeline_status.txt
tail -n 60 scripts/experiments/setup/nm_single_source_downstream/run_logs/pipeline.log
```

The pipeline phases are `resolve → evaluate → assemble → plot`.
`ONLY=<phase>` reruns one phase. `DRY_RUN=1` on `run_eval_sweep.sh` prints
commands without launching them.

## Regression floors: Ukraine-suspended and twibot20

`ukr_rus_suspended` has the three core profile columns in its source
`user_data.csv`, but its canonical graph stores only the suspended-account
classification label. The baseline runner creates an experiment-local enriched
copy under this worktree's ignored `state/`; it does not overwrite the canonical
graph in `/dataMeR1/phil/data`.

The runner evaluates three matched 10-shot floors on Ukraine-suspended and
twibot20 over five seeds by default (`SEEDS=0,1,2,3,4`):

- `raw_features`: episodic Ridge on raw 768-d bio embeddings;
- `raw_degree`: episodic Ridge on directed structural features;
- `random_init`: the normal frozen encoder/readout with untrained weights.

```bash
tmux new-session -d -s nmssd_regbase \
  'export PATH="/home/mhchu/miniconda3/bin:$PATH"; \
   EVAL_GPUS="0,1" \
   bash scripts/experiments/setup/nm_single_source_downstream/run_regression_baselines_tucker.sh'
```

The three targets are followers, statuses, and account age, with `log1p`,
10 support nodes and 12 queries. Each seed resamples the support/query episodes
for every floor; for `random_init` it also changes the encoder initialization.
The experiment-only episode-seed offset defaults to zero everywhere else, so
the repository's historical fixed eval episodes are unchanged.

## Deliverables

Under `analysis/transfer/matrices/prodigy_nm/downstream/nm_single_source_downstream/`:

- `data/model_manifest.csv` — exact checkpoint provenance;
- `data/classification.csv` — 8 × 4 ROC-AUC matrix plus row mean;
- `data/regression.csv` — 8 × 12 Spearman matrix plus row mean;
- `data/regression_by_dataset.csv` — regression averaged over three targets;
- `data/results_long.csv` — tidy source/task/dataset/target results;
- `figures/single_source_downstream_heatmaps.{png,pdf}` — annotated
  classification and regression heatmaps.
- `data/regression_baseline_seeds.csv` — all per-seed floor scores;
- `data/regression_baselines.csv` and `figures/regression_baselines.{png,pdf}` —
  mean ± sample SD for the matched Ukraine-suspended/twibot20 regression floors.
