# Final-core PRODIGY protocol

This directory registers the clean final-core PRODIGY experiment. Training and
evaluation are intentionally separated so code changes made after launch cannot
alter a live process.

## Training contract

- 31 physically distinct source sets: nine specialists plus the deduplicated
  rungs of frozen orders A/B/C;
- seeds 0, 1, and 2 (93 runs);
- exactly 2,500 optimizer updates, batch size 4, balanced source sampling;
- checkpoints after 100, 300, 900, and 2,500 completed updates;
- one immutable all-nine artifact split by unordered edge pair into 70% train,
  15% validation, and 15% test with seed 0;
- train positives and message passing use `static_train`; validation and test
  positives use disjoint `static_validation` and `static_test` views.

`build_split_artifact_tucker.py` refuses to overwrite an artifact and validates
the exact nine-source registry. `run_training_tucker.sh` uses a dedicated state
and log root and never resumes an ambiguous partial run.

## Final fixed-test matrix and ladder evaluation

`run_evaluation_tucker.sh` and its validation-selected 93-row result are retained
only as provenance. They are not the final evaluation: that queue interpreted
`test_len_cap=500` as 500 episodes even though it meant 500 batches, selected a
checkpoint on validation, and restricted each model's test stream to its own
training sources.

The replacement is `run_fixed_test_tucker.sh`. Its contract is:

- only `state_dict_2500.ckpt`; no validation dataloader and no checkpoint selection;
- 512 fixed `static_test` episodes per target and cell, with message passing on
  `static_train` only;
- batch size 32 (16 full batches); batch size 64 is forbidden for this artifact
  because the concurrent smoke reproducibly triggered invalid CUDA indices;
- 3 seeds x 31 physical models x 9 individual targets = 837 unique cells;
- one graph load per persistent worker, two workers on each owned GPU 0--3;
- target-major materialization: each worker collates one target's 16 CPU batches
  once, replays fresh clones across its assigned checkpoints, and releases that
  cache before materializing the next target;
- per-target raw and observed episode-stream fingerprints that must agree across
  all models, training seeds, and persistent workers;
- strict 243-cell specialist matrix, 675-cell physical ladder, and 729-row
  alias-expanded ladder outputs. The 81 matrix/rung-1 cells and shared rung 9 are
  evaluated physically only once.

Run this only from its isolated descendant worktree. Point it at the completed
training state root; never switch or pull the training worktree.

```bash
tmux new-session -d -s finalcore_fixed_test \
  'export PATH="/home/mhchu/miniconda3/bin:$PATH"; \
   export TRAINING_STATE_ROOT=/dataMeR1/phil/gfm/prodigy-final-core/state/final_core; \
   bash scripts/experiments/setup/final_core/run_fixed_test_tucker.sh'
```

The queue is resumable: completed physical cells are validated before being
skipped. Raw evidence and strict matrices/tables are under
`log/final_core_fixed_test/production/bs{64,32}/`.

## Complete specialist ROC-AUC matrix

The first strict aggregate retained `score`/accuracy but not the multiclass
ROC-AUC already computed inside `TrainerFS`. Because logits were not retained,
missing AUC cells cannot be reconstructed offline. The dedicated AUC queue
replays the same frozen test protocol for only the 27 specialist checkpoints
(3 seeds x 9 sources x 9 targets = 243 cells), and atomically stores accuracy,
macro F1, and macro one-vs-rest ROC-AUC in every result JSON. Both the raw plan
and observed-stream hashes must match the published fixed-test fingerprint
ledger before any result is accepted.

Run it from the isolated `experiment/final-core-auc-grid` worktree after owned
GPUs 0--3 are free. Its resource gate waits rather than competing with an
existing job:

```bash
tmux new-session -d -s finalcore_auc \
  'export PATH="/home/mhchu/miniconda3/bin:$PATH"; \
   cd /dataMeR1/phil/gfm/prodigy-final-core-auc; \
   bash scripts/experiments/setup/final_core/run_auc_matrix_tucker.sh'
```

The final mean matrix is
`log/final_core_auc/production/bs32/summary/single_source_roc_auc_ovr_macro_three_seed_mean.csv`.

For a replacement smoke, set `SMOKE_ONLY=1`; it evaluates two checkpoints per
worker so replay is exercised, then exits. After that passes, production can
resume an earlier result tree by setting `SKIP_SMOKE=1` and
`PRODUCTION_RESULTS_ROOT=/absolute/path/to/results`. Existing cells must match
the frozen plan and observed stream fingerprints before they are accepted.
