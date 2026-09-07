# TRACE schedule scaling

This experiment tests the old ladder observation that blocked sequential exposure
helps at two sources but degrades as mixtures grow. It is a controlled rungs 2/3/4,
three-seed comparison of:

- `blocked`: one contiguous block per source;
- `replay100`: cyclic 100-episode source blocks;
- `interleaved`: cyclic one-episode blocks.

Within every rung and seed, all three arms consume exactly the same per-source episode
multiset. Independent per-source Python and Torch RNG streams make schedule order the
only data intervention. Seed-wise rotations vary which source comes last. Exact
consumed episodes and per-node anchor/support/query counts are retained.

On Tucker, use the dedicated `codex/trace-schedule-scaling` worktree. First run:

```bash
PHASE=smoke GPUS="0" bash scripts/experiments/setup/trace_schedule_scaling/run_tucker.sh
```

Then launch the 27 full arms on the owned GPUs:

```bash
tmux new-session -d -s tracesched \
  'export PATH="/home/mhchu/miniconda3/bin:$PATH"; PHASE=full GPUS="0 1 2 3" bash scripts/experiments/setup/trace_schedule_scaling/run_tucker.sh'
```

After training completes, replay the terminal checkpoints on five held-out targets and
two disjoint fixed episode streams:

```bash
tmux new-session -d -s tracesched-replay \
  'export PATH="/home/mhchu/miniconda3/bin:$PATH"; GPUS="0 1 2 3" bash scripts/experiments/setup/trace_schedule_scaling/run_replay_tucker.sh'
```

The primary estimand is paired target AUC/accuracy by schedule within rung and seed.
The mechanistic estimand is whether support-only TRACE health changes before and with
held-out performance, and whether `replay100` removes the two-to-many source reversal.
