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

## Frozen checkpoint selection and test

Evaluation has a hard phase barrier:

1. Every checkpoint is evaluated on 500 fixed validation episodes. The highest
   validation score is selected; an exact tie chooses the earlier checkpoint.
2. Only after all 93 `selection.json` files exist does the queue create the
   validation-complete marker.
3. The test phase reads each frozen selection and evaluates that checkpoint once
   on 500 fixed `static_test` episodes. It never compares checkpoints on test.

Validation-only evaluation does not construct or iterate a test dataloader.
`evaluate_model.py` loads the large graph once and reuses it for all four
validation checkpoints. The Tucker default uses two processes per H100 (eight
concurrent graph loads). Each measured process uses about 118 GiB host RAM but
only about 3.3 GiB VRAM, so the queue enforces a host-memory reserve before it
starts. Override with `SLOTS_PER_GPU=1` on a host with less available RAM.

Run this from a separate descendant worktree after training completes. Point it
at the state root of the worktree that is already training; do not switch or pull
that live worktree.

```bash
export TRAINING_STATE_ROOT=/dataMeR1/phil/gfm/prodigy-final-core/state/final_core
DRY_RUN=1 bash scripts/experiments/setup/final_core/run_evaluation_tucker.sh

tmux new-session -d -s finalcore_eval \
  'export PATH="/home/mhchu/miniconda3/bin:$PATH"; \
   export TRAINING_STATE_ROOT=/dataMeR1/phil/gfm/prodigy-final-core/state/final_core; \
   bash scripts/experiments/setup/final_core/run_evaluation_tucker.sh'
```

The queue is resumable: completed selections and tests are validated and skipped.
Raw per-seed evidence is under `log/final_core_eval/results`; strict aggregation
writes the 372-row validation trajectory, 93-row physical test grid, alias-expanded
grid, and 31-row three-seed summary under `log/final_core_eval/summary`.
