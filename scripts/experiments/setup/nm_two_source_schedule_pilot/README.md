# Two-source sequential-versus-interleaved pilot

This six-model pilot tests whether blocked sequential source presentation can be worse
than balanced interleaving even with only two pretraining graphs. It covers two pairs:

- COVID-19 Twitter with COVID Political;
- COVID-19 Twitter with CP-HK Twitter.

Each pair has a balanced-interleaved arm and both sequential orders. All models use seed
0 and the recent final-core training contract: the split-seed-0 all-nine artifact,
background-only two-hop neighbor matching, fanouts `9,9`, 101-node cap, 30-way/3-shot/
4-query episodes, and 2,500 optimizer updates. The source scheduler requires batch size
1, so all six arms use batch size 1. Interleaved arms receive 1,250 expected episodes per
source; sequential arms receive exactly 1,250 episodes from each source in one block.

The primary endpoint is 10-shot node-classification ROC-AUC on COVID Political, Election
2020, Ukraine Suspended, and TwiBot-20. Step 1,250 is saved so the sequential transition
can be diagnosed later, but the registered schedule comparison uses step 2,500.

## Validate and launch on Tucker

Use a dedicated worktree and confirm no other job is using it before pulling. The launcher
uses GPUs 0--3 by explicit authorization and maintains one queue per GPU: four models start
immediately, and the remaining two start when their assigned GPUs become free.

```bash
python scripts/experiments/setup/nm_two_source_schedule_pilot/validate_plan.py --check-data

DRY_RUN=1 GPUS="0 1 2 3" \
  bash scripts/experiments/setup/nm_two_source_schedule_pilot/run_tucker.sh

tmux new-session -d -s nm2sched \
  'export PATH="/home/mhchu/miniconda3/bin:$PATH"; \
   GPUS="0 1 2 3" bash scripts/experiments/setup/nm_two_source_schedule_pilot/run_tucker.sh \
   > log/nm_two_source_schedule_pilot/orchestrator.log 2>&1'
```

Training checkpoints are written beneath `state/nm_two_source_schedule_pilot/`; training,
evaluation, model-list, and provenance logs are beneath `log/nm_two_source_schedule_pilot/`.
The launcher refuses incomplete same-stamp directories and skips complete terminal
checkpoints, so a fixed `RUN_STAMP` can be resumed safely.
