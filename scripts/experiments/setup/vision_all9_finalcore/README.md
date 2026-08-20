# VISION all-nine final-core-compute run

This experiment trains one seed-0 VISION model with its native label-free
feature-similarity pseudo-episode objective on the exact nine-source final-core
mixture. Each pseudo-episode is confined to one source, and sources are sampled
uniformly and independently. The budget matches final-core: 2,500 optimizer
updates, batch size 4, and therefore 10,000 pseudo-episodes. Checkpoints are saved
at updates 100, 300, 900, and 2,500.

The 107 GB merged graph remains in host memory. Adaptive VISION task features are
computed on CPU, source-restricted pseudo-tasks are selected against those
features, and only sampled episode subgraphs are transferred to the GPU. The
terminal checkpoint is evaluated tuning-free on the five registered downstream
classification targets, including Facebook pages.

Run from an isolated Tucker worktree on an owned free GPU:

```bash
tmux new-session -d -s vision-all9-finalcore \
  'export PATH="/home/mhchu/miniconda3/bin:$PATH"; \
   cd /dataMeR1/phil/gfm/prodigy-vision-all9; \
   GPU=0 bash scripts/experiments/setup/vision_all9_finalcore/run_tucker.sh'
```

The queue refuses to start on an occupied GPU and refuses ambiguous partial
training state. Outputs live under `state/vision_all9_finalcore/` and
`log/vision_all9_finalcore/` in that worktree.
