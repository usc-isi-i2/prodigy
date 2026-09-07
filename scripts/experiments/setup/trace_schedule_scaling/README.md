# TRACE schedule scaling

This experiment tests the old ladder observation that blocked sequential exposure
helps at two sources but degrades as mixtures grow. It is a controlled rungs 2/3/4,
three-seed comparison of:

- `blocked`: one contiguous block per source;
- `replay100`: cyclic 100-episode source blocks;
- `interleaved`: cyclic one-episode blocks.

Within every rung and seed, all three arms consume exactly the same per-source episode
multiset. Independent per-source Python and Torch RNG streams make schedule order the
only data intervention, including stochastic k-hop context expansion in loader
workers. Seed-wise rotations vary which source comes last. Exact consumed anchors,
members, support/query roles, source IDs, context-node order, and context edges are retained. Dense
graph-wide role counters are disabled because they would bloat every checkpoint on
the tens-of-millions-node merge; the consumed-episode record is both smaller and more exact for
this comparison.

On Tucker, use the dedicated `codex/trace-schedule-scaling` worktree. First run:

```bash
PHASE=smoke GPUS="0" bash scripts/experiments/setup/trace_schedule_scaling/run_tucker.sh
```

Then launch the 27 full arms on the owned GPUs:

```bash
tmux new-session -d -s tracesched \
  'export PATH="/home/mhchu/miniconda3/bin:$PATH"; PHASE=full GPUS="0 1 2 3" bash scripts/experiments/setup/trace_schedule_scaling/run_tucker.sh'
```

Verify terminal checkpoints, saved contracts, exact source order, and per-source
full-graph episode identity before evaluation:

```bash
python -m scripts.experiments.setup.trace_schedule_scaling.verify_training \
  --state-root state/trace_schedule_scaling \
  --log-root log/trace_schedule_scaling \
  --run-stamp 20260906v1 \
  --output log/trace_schedule_scaling/launch/verification_20260906v1.json
```

After training completes, replay the terminal checkpoints on five held-out targets and
two disjoint fixed episode streams:

```bash
tmux new-session -d -s tracesched-replay \
  'export PATH="/home/mhchu/miniconda3/bin:$PATH"; GPUS="0 1 2 3" bash scripts/experiments/setup/trace_schedule_scaling/run_replay_tucker.sh'
```

Convert the verified replay shards into one TRACE analysis record per target:

```bash
python -m scripts.experiments.setup.trace_schedule_scaling.export_replays \
  --replay-root log/trace_schedule_scaling/replay_20260906v1 \
  --output log/trace_schedule_scaling/analysis_inputs_20260906v1
```

The primary estimand is paired target AUC/accuracy by schedule within rung and seed.
The mechanistic estimand is whether support-only TRACE health changes before and with
held-out performance, and whether `replay100` removes the two-to-many source reversal.
