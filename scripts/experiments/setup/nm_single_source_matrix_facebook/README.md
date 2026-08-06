# Facebook extension of the NM single-source matrix

This adds the ninth source/target to the historical matched-40k, one-hop,
30-way/3-shot neighbor-matching matrix.

The run creates 17 new matrix cells:

- the eight existing Twitter specialists evaluated on the structural Facebook graph;
- the Facebook-only specialist evaluated on all nine graphs.

The Facebook specialist is trained from random initialization. The earlier 2k CARC
smoke checkpoint is not reused. Training uses the 119,228-node structural view so
isolated attributed pages cannot enter neighbor-matching sampling.

## Tucker command

After pulling this branch into the idle `prodigy-facebook` worktree, launch the
pipeline in detached tmux:

```bash
tmux new-session -d -s nm-ss-facebook \
  'export PATH="/home/mhchu/miniconda3/bin:$PATH"; \
   bash scripts/experiments/setup/nm_single_source_matrix_facebook/run_matrix_tucker.sh'
```

The pipeline refuses to start unless owned GPUs 0-3 are idle and all eight historical
`state_dict_40000.ckpt` files are present in the main Tucker checkout. Runtime model
lists, status, and logs are written beneath the setup folder's ignored `run_logs/`.

Use `DRY_RUN=1 bash .../run_matrix_tucker.sh` to validate those inputs without
starting training or evaluation.

Phase 1 trains Facebook on GPU 0 while GPUs 1-3 evaluate the old specialists on
Facebook. Phase 2 evaluates the new Facebook specialist on all nine targets using
GPUs 0-3.
