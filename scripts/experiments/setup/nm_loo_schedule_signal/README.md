# LOO exposure x source-block signal experiment

This is the deliberately minimal 2 exposure x 3 schedule x 3 seed experiment.
It uses the recent final-core LOO mixture that holds out `ukr_rus`: COVID is the
largest retained source (23,012,850 nodes), Ukraine-Suspended the smallest
(72,295), a 318.32x ratio. The eight retained sources total 24,231,447 nodes;
COVID has 94.97% of nodes. Counts come from `docs/graph_catalog.json` verified
2026-09-04.

The only departure from the final-core LOO training YAML is `batch_size: 1`,
required by the explicit source scheduler so a scheduled episode equals an
optimizer step. Architecture, features, two-hop 9/9 sampler, 101-node cap,
30-way/3-shot/4-query task, learning rate, weight decay, 2,500 updates, and
checkpoint/evaluation settings are unchanged.

`make_configs.py` deterministically predeclares exactly 18 configs. `k=1` and
`k=16` use seeded schedules with exact largest-remainder exposure quotas;
`blocked` presents each source once, in a seeded order, with the same quota.
The proportional quota follows source node counts; uniform assigns 312 or 313
steps to each graph. No additional block sizes or seeds are generated.

Launch on idle owned GPUs 2 and 3:

```bash
DRY_RUN=1 GPUS="2 3" bash scripts/experiments/setup/nm_loo_schedule_signal/run_tucker.sh
tmux new-session -d -s nm_loo_signal 'export PATH="/home/mhchu/miniconda3/bin:$PATH"; cd /dataMeR1/phil/gfm/prodigy-loosignal; GPUS="2 3" bash scripts/experiments/setup/nm_loo_schedule_signal/run_tucker.sh > log/nm_loo_schedule_signal_orchestrator.log 2>&1'
```

Evaluate every terminal checkpoint on the same frozen 512-episode held-out
`ukr_rus` NM stream used by `nm_leave_one_out_finalcore`; use the existing
`evaluate_loo.py` protocol after forming a model list from the shared manifest.
That archived stream requires `neighbor_matching_member_policy: lowest_sorted`;
the evaluation config must set it explicitly on current code.
