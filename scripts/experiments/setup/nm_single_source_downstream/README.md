# NM single-source downstream matrix

## Question

Do the eight matched-40k single-source neighbor-matching (NM) encoders differ in
their transfer to downstream node classification and node regression?

This complements `analysis/nm_single_source_matrix`, which evaluates the same
source models on the NM pretraining task. The outputs here are downstream-task
matrices:

- node classification: 8 source models × 4 labeled graphs, 10-shot ROC-AUC;
- node regression: 8 source models × 4 profile graphs × 3 targets, 10-shot
  Spearman with `log1p` targets.

## Checkpoint policy

All scores use `state_dict_40000.ckpt`. Three original matched-40k checkpoints
still exist in the main Tucker checkout and are reused:

| source | model key |
|---|---|
| Ukraine | `ukr_only_nm` |
| COVID | `nm_ss_covid19_twitter` |
| Election 2020 | `nm_ss_election2020` |

The other five original checkpoints were no longer present on Tucker, so this
experiment retrains them from the original single-source configs:

`midterm`, `covid_political`, `ukr_rus_suspended`, `twibot20`,
`cp_hk_twitter`.

Those configs keep the original 5 × 10k schedule and explicitly select the 40k
checkpoint. With the post-2026-07-26 trainer they also emit a terminal 50k
checkpoint; it is deliberately ignored.

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
   TRAIN_GPUS="0 1" EVAL_GPUS="0,1" \
   bash scripts/experiments/setup/nm_single_source_downstream/run_pipeline_tucker.sh'
```

Progress:

```bash
cat scripts/experiments/setup/nm_single_source_downstream/run_logs/pipeline_status.txt
tail -n 60 scripts/experiments/setup/nm_single_source_downstream/run_logs/pipeline.log
```

The pipeline phases are `train → resolve → evaluate → assemble → plot`.
`ONLY=<phase>` reruns one phase. `DRY_RUN=1` on `run_training.sh` or
`run_eval_sweep.sh` prints commands without launching them.

## Deliverables

Under `analysis/nm_single_source_downstream/`:

- `data/model_manifest.csv` — exact checkpoint provenance;
- `data/classification.csv` — 8 × 4 ROC-AUC matrix plus row mean;
- `data/regression.csv` — 8 × 12 Spearman matrix plus row mean;
- `data/regression_by_dataset.csv` — regression averaged over three targets;
- `data/results_long.csv` — tidy source/task/dataset/target results;
- `figures/single_source_downstream_heatmaps.{png,pdf}` — annotated
  classification and regression heatmaps.
