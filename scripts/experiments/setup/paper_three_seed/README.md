# Paper three-seed replication

This queue materializes the two missing PRODIGY training seeds required by the
LoG extended-abstract design. It runs exactly 57 models per seed:

- eight one-hop GraphSAGE specialists;
- 18 distinct multi-source mixtures across the three one-hop source orders;
- eight topical-order two-hop GraphSAGE rungs;
- eight topical-order one-hop GATv2 rungs;
- 15 fixed-exposure two-hop rungs (orders A and C, shared endpoint once).

The queue uses deterministic run names beneath its own experiment worktree and
skips a model only when its 40k comparison checkpoint already exists. The
fixed-exposure configurations retain their registered `10k × source_count`
budgets; every other family is pinned to exactly 40k optimizer steps.

On Tucker, after confirming owned GPUs 0-3 are free:

```bash
DRY_RUN=1 SEEDS="1 2" GPUS="0 1 2 3" bash scripts/experiments/setup/paper_three_seed/run_queue_tucker.sh
tmux new-session -d -s prodigy-paper-seeds \
  'export PATH="/home/mhchu/miniconda3/bin:$PATH"; SEEDS="1 2" GPUS="0 1 2 3" bash scripts/experiments/setup/paper_three_seed/run_queue_tucker.sh > log/paper_three_seed/queue.log 2>&1'
```

After training completes, the queue evaluates all checkpoints on the eight fixed
NM receiver episode sets. One-hop GraphSAGE, GATv2, and two-hop GraphSAGE are
evaluated separately so that each checkpoint is reconstructed with the matching
architecture and sampler.
