# PRODIGY ladder cross-task evaluation

This evaluation-only experiment fills the two missing cells needed to separate
training budget from evaluation task:

1. evaluate the seed-0, step-100 architecture-matrix ladder on the exact final-core
   fixed native neighbor-matching streams; and
2. evaluate the three-seed, step-2,500 final-core ladder on the exact four fixed
   downstream binary-classification streams used by the architecture matrix.

No model is retrained and no validation or checkpoint selection occurs. The step-100
NM grid contains 25 distinct physical ladder checkpoints x 9 targets = 225 cells. The
step-2,500 downstream grid contains 3 seeds x 25 physical ladder checkpoints x 4
targets = 300 cells. Aggregation expands shared rung-1/rung-9 checkpoints to the three
logical orders. Both sweeps verify their episode fingerprints against the already
published evaluation streams, and both can resume validated partial results.

The Tucker launcher uses only GPUs 0 and 1. It permits coexistence only while each GPU
uses at most 12 GiB before launch and at least 700 GiB host RAM remains available. It
does not alter either archived training worktree.

```bash
export PATH="/home/mhchu/miniconda3/bin:$PATH"
tmux new-session -d -s ladder-cross-task \
  'export PATH="/home/mhchu/miniconda3/bin:$PATH"; \
   cd /dataMeR1/phil/gfm/prodigy-ladder-xeval; \
   bash scripts/experiments/setup/ladder_cross_task_eval/run_tucker.sh \
   > log/ladder_cross_task_eval/orchestrator.log 2>&1'
```
