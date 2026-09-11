# Pure node-feature MLP source ladder

One nested Social-9 source order (the existing `SOURCE_ORDER`), seed 0,
2,500 optimizer updates per rung, nine independently initialized models.
Every update uses one source, round-robin; positives and uniform negatives stay
inside that source. The largest difference between per-source update counts is one.
The architecture, optimizer, 1,024-positive batch size and five negatives per positive
come from `configs/node_only_transfer.yaml`.

This reuses the fast `DirectLinkLoader` from the node-only transfer experiment.
Features stay on GPU when they fit in a conservative 70 GiB budget (with at least
6 GiB currently free memory reserved); smaller sources are prioritized. Other
sources use the same endpoint-only pinned CPU batch path. No neighborhood sampling
or message passing is performed. Each worker loads each CPU graph once and reuses it
across its assigned rungs. Only Tucker GPUs 2 and 3 are used.

Evaluate each terminal checkpoint against all six existing LP targets with the
canonical cached edge partition, degree-matched negatives, and validation-selected
cosine orientation. `best.pt` is an evaluator compatibility link to the fixed terminal
checkpoint; validation loss is diagnostic and never selects the model. The output
has 54 cells, with explicit `target_in_pretraining` flags. Included targets still use
held-out edges. This is a nested source ladder, not a leave-target-out experiment;
do not average changing unseen-target membership into a scaling claim. The older
GraphSAGE seven-target classification ladder has a different evaluation protocol.

Run `bash scripts/run_node_mlp_ladder_tucker.sh` in a dedicated Tucker worktree
inside tmux. Override `STATE_ROOT`, `RESULTS_ROOT`, `LOG_ROOT`, or `CACHE_ROOT`
as needed. Existing log directories and partial training runs are refused. Complete
models can be skipped with a fresh log directory. Data and checkpoints remain under
`/dataMeR1/phil/gfm/mixture-scaling`; existing single-source results stay separate.

The command's `plan` phase is read-only. The `aggregate` phase checks completion,
source sets, checkpoint budget and LP gates before writing `ladder_results.csv`
and `COMPLETE.json`. No run should be called complete before those outputs exist.

## W&B tracking

Each rung creates one W&B run, offline by default even when invoked directly in
Python. Override `WANDB_MODE` or `--wandb-mode` explicitly for online or disabled
mode. The default project is `node-mlp-ladder`; `WANDB_PROJECT` and
`WANDB_RUN_GROUP` are supported. There is no automatic upload or login.

Every 25 updates, log example-weighted window BCE and per-source window BCE,
example counts, optimizer step and elapsed time. `--log-interval` changes the
window size. Every batch contributes, including the final partial window. Final
per-source validation losses are diagnostic, not a validation trajectory.
Configuration, source sets, source update counts and timings are recorded too.
The human-readable `metrics.jsonl` mirrors the metrics in W&B. Offline records
live at `<state-root>/lp/<rung>/wandb/offline-run-*`; `wandb_run.json` identifies
each run. They can later be uploaded explicitly using `wandb sync <offline-run-dir>`.

Use fresh state/results/log roots for a rerun, since completed rungs are skipped
and partial runs are refused. Tracking does not change the fixed training budget.
