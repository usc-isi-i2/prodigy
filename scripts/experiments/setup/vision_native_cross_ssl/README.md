# VISION native cross-SSL matrix

This experiment evaluates the five existing seed-0 VISION native specialists
on the native label-free feature-similarity objective itself. It is distinct
from downstream CLS: no target labels or classification splits enter these
pseudo-tasks.

For each source specialist, checkpoints 20/60/100/300/900 are evaluated on 128
deterministic feature-similarity pseudo-episodes from each of the same five
registered target graphs. Every target's support/query node fingerprint must be
identical across all sources and checkpoints. The outputs record native SSL
loss, pseudo-classification accuracy, cross-entropy, and contrastive loss. This
is a fixed-compute seed-0 5×5 registered panel, not a convergence comparison or
a nine-graph final-core matrix.

Run from an isolated Tucker worktree using only owned GPUs 2 and 3:

```bash
tmux new-session -d -s vision-cross-ssl \
  'export PATH="/home/mhchu/miniconda3/bin:$PATH"; \
   cd /dataMeR1/phil/gfm/prodigy-native-matrix; \
   GPU_A=2 GPU_B=3 bash \
   scripts/experiments/setup/vision_native_cross_ssl/run_tucker.sh'
```
