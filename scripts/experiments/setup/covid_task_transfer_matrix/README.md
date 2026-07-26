# COVID Task-Transfer Matrix

This folder evaluates a task-transfer matrix on the COVID retweet graph only.

Rows are the training task:

- `nm`: neighbor matching
- `cl`: contrastive same-node view matching
- `fp`: masked feature prediction

Columns are the eval task over the same COVID graph. The intended first matrix
uses the best checkpoints from the existing single-task runs:

```text
state/task_transfer_covid_nm_smoke_<timestamp>/state_dict
state/task_transfer_covid_cl_smoke_<timestamp>/state_dict
state/task_transfer_covid_fp_smoke_<timestamp>/state_dict
```

Defaults target `/dataMeR1`, matching the current experiment state.

## Relationship to Earlier Matrix Experiments

This follows the same local structure as:

- `scripts/experiments/nm_cross_source_shortcut`
- `scripts/experiments/nm_transfer_matrix`
- `scripts/experiments/nm_covid_midterm`

Those experiments build dataset-transfer matrices for the NM task. This one
instead holds the dataset fixed to COVID and varies the train/eval task.

The existing shared eval runner, `scripts/eval/eval_ckpts_all_graph_tasks_tucker.py`,
only supports `nm/lp/pl`, so this folder uses a dedicated launcher for
`nm/cl/fp`.

## Protocols

### Zero-Adaptation Compatibility

This is the quick diagnostic matrix already used first. It loads each checkpoint
and immediately evaluates it on every task:

```bash
cd /dataMeR1/phil/gfm/prodigy/scripts/experiments/covid_task_transfer_matrix

STATE_DIR=/dataMeR1/phil/gfm/prodigy/state ./make_model_list.sh

DRY_RUN=1 ./eval_task_matrix_tucker.sh
./eval_task_matrix_tucker.sh

python3 build_task_matrix.py \
  --protocol zero \
  --log-root /dataMeR1/phil/gfm/prodigy/log \
  --out-csv task_matrix_zero.csv
```

This is not a fair transfer matrix for FP because `NM -> FP` and `CL -> FP`
evaluate with a newly initialized FP reconstruction head.

### Fixed-Adaptation Transfer

This is the fairer transfer protocol. Each cell loads the source checkpoint,
trains on the eval task for the same number of updates, then evaluates once at
the final step. A `scratch` row is included as a baseline.

```bash
cd /dataMeR1/phil/gfm/prodigy/scripts/experiments/covid_task_transfer_matrix

# 1. Build rows from the latest best checkpoints.
STATE_DIR=/dataMeR1/phil/gfm/prodigy/state ./make_model_list.sh

# 2. Preview the 4x3 adaptation commands.
ADAPT_STEPS=1000 DRY_RUN=1 ./adapt_task_matrix_tucker.sh

# 3. Run fixed-budget adaptation. Use GPUS to parallelize if desired.
ADAPT_STEPS=1000 GPUS=0,1,2 ./adapt_task_matrix_tucker.sh

# 4. Build the matrix.
python3 build_task_matrix.py \
  --protocol adapt \
  --log-root /dataMeR1/phil/gfm/prodigy/log \
  --out-csv task_matrix_adapt_1000.csv
```

Set `GPUS=0,1,2` to run up to three eval jobs in parallel. With no `GPUS`, the
scripts run sequentially on `DEVICE`.

## Notes

- NM/CL cells report classifier-style metrics from `metrics_test_step0.json`
  (`accuracy`, `f1`, `roc_auc`).
- FP cells report the scalar feature-prediction score from
  `scores_test_step0.json`. This is negative MSE, so higher is better.
- In the fixed-adaptation protocol, every eval task gets the same adaptation
  budget, so `NM/CL -> FP` train the FP `aux_header` before scoring.
