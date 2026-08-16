# Final-core step-2,500 classification ladders

Evaluate the 25 distinct physical PRODIGY checkpoints behind the three nine-rung
final-core orders, for all three training seeds, on five fixed downstream
classification streams. The protocol is 128 fixed 2-way/10-shot episodes per target.

The complete sweep is 3 seeds × 25 checkpoints × 5 targets = 375 physical evaluation
cells (48,000 episodes). Rung-1 specialists and the shared all-nine checkpoint are
evaluated once per seed and can later be expanded into the 3 × 3 × 9 logical ladder.

On Tucker, from this experiment's dedicated worktree:

```bash
tmux new-session -d -s finalcore-cls2500 \
  'export PATH="/home/mhchu/miniconda3/bin:$PATH"; \
   bash scripts/experiments/setup/finalcore_cls2500_ladders/run_tucker.sh'
```

Inspect progress with:

```bash
tail -f log/finalcore_cls2500_ladders/queue/seed0_gpu0.log
```

The launcher uses only GPUs 0 and 1, runs one worker per GPU, completes seeds in order,
refuses to overwrite result shards, and writes the validated aggregate to
`log/finalcore_cls2500_ladders/classification_long.tsv`.

Dry-run command construction with `DRY_RUN=1 bash .../run_tucker.sh` on Tucker.
