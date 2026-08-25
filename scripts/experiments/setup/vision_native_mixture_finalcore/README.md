# VISION native mixture diversity at final-core compute

This experiment fills the missing VISION mixture-diversity→downstream-CLS family
using only VISION's native label-free feature-similarity pseudo-episode
objective. It reuses the three registered final-core source orders A/B/C and
trains their odd rungs 1/3/5/7/9. Deduplication leaves 13 unique source sets;
the existing all-nine seed-0 checkpoint is reused, so 12 new models train.

Every new model uses the same fixed-compute contract as the registered all-nine
VISION model: seed 0, 2,500 optimizer updates, batch size 4 (10,000
pseudo-episodes), uniform source sampling, and checkpoints at
100/300/900/2,500. Every checkpoint is evaluated tuning-free on the same five
fixed downstream CLS episode streams. The three orders provide composition
replication at each mixture size; this is a fixed-compute study, not a
convergence-trained comparison.

Run from an isolated Tucker worktree on owned GPUs 2 and 3:

```bash
tmux new-session -d -s vision-native-mixture \
  'export PATH="/home/mhchu/miniconda3/bin:$PATH"; \
   cd /dataMeR1/phil/gfm/prodigy-vision-mixture; \
   GPU_A=2 GPU_B=3 bash \
   scripts/experiments/setup/vision_native_mixture_finalcore/run_tucker.sh'
```

Outputs are worktree-local under `state/vision_native_mixture_finalcore/` and
`log/vision_native_mixture_finalcore/`. The launcher refuses GPUs other than 2
or 3 and preserves every checkpoint's downstream trajectory.
