# Native-model result-matrix overnight queue

This wrapper executes the remaining feasible work in the requested priority
order, without allowing GPU overlap:

1. the matched frozen-encoder adaptation grid for PRODIGY, VISION, SAMGPT,
   GraphSAGE, raw logistic, and raw MLP, plus the separate GraphSAGE prefix
   saturation grid;
2. the checkpoint-only VISION native cross-SSL replay;
3. the genuinely missing VISION native mixture-diversity training and full
   downstream checkpoint replay.

Each child experiment owns its state, logs, protocol checks, and completion
marker. The wrapper is resumable at child-experiment boundaries and refuses to
start if physical Tucker GPU 2 or 3 is occupied. No command selects GPUs 0, 1,
or 4–7.

Launch only from the isolated synced result-matrix worktree:

```bash
tmux new-session -d -s native-matrix-overnight \
  'export PATH="/home/mhchu/miniconda3/bin:$PATH"; \
   cd /dataMeR1/phil/gfm/prodigy-native-matrix; \
   bash scripts/experiments/setup/native_model_result_matrix_overnight/run_tucker.sh'
```

The parent completion marker is
`log/native_model_result_matrix_overnight/COMPLETE`. A missing marker identifies
the first unfinished child without hiding its underlying log.
