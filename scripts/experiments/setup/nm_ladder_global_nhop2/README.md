# Naive-global two-hop NM ladder

This is the deliberate failure-mode comparison for the canonical Order A fair-two-hop
ladder. Each rung treats its growing disjoint merge as one graph. It omits all
`graph_id`, source-subset, and source-balancing settings, allowing mixed-source NM
episodes and weighting sources by eligible-node mass. Everything else matches
`setup/nm_ladder_nhop2`.

Rung 1 is not retrained: a single source makes global and within-source sampling
identical. Rungs 2–8 are independent seed-0, from-scratch, honest-40k runs.

## Streaming GPU protocol

Only Tucker GPUs 0 and 1 are allowed. GPU 0 trains all seven rungs sequentially. GPU 1
watches for stable checkpoints and evaluates 10k/20k/30k on a sentinel panel (Ukraine,
the newcomer, and Hong Kong) and 40k on all eight targets. Evaluation queues if it falls
behind and never shares a GPU with training.

From this experiment's dedicated Tucker worktree:

```bash
bash scripts/experiments/setup/nm_ladder_global_nhop2/check_inputs_tucker.sh

tmux new-session -d -s nmglobal-eval \
  'export PATH="/home/mhchu/miniconda3/bin:$PATH"; \
   GPU=1 bash scripts/experiments/setup/nm_ladder_global_nhop2/watch_and_eval_tucker.sh \
   > scripts/experiments/setup/nm_ladder_global_nhop2/run_logs/watcher.log 2>&1'

tmux new-session -d -s nmglobal-train \
  'export PATH="/home/mhchu/miniconda3/bin:$PATH"; \
   GPU=0 bash scripts/experiments/setup/nm_ladder_global_nhop2/run_all_train_tucker.sh \
   > scripts/experiments/setup/nm_ladder_global_nhop2/run_logs/orchestrator.log 2>&1'
```

The primary result is always the paired terminal-40k difference against the existing
interleaved fair-two-hop Order A ladder. Intermediate evaluations are monitoring and
trajectory evidence, not substitutes for the terminal comparison.
