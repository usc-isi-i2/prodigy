# Order A rung 9: add Facebook to the NM ladder

This appends `facebook-page-reference` to the published eight-source Order A ladder.
The rung-9 model is trained from random initialization on the disjoint all-nine merge,
using the same one-hop, within-source balanced, matched-40k protocol as rungs 1–8.

The pipeline creates the complete 9×9 ladder extension in two parallel phases:

1. Train Order A rung 9 on GPU 0 while evaluating historical rungs 1–8 on Facebook
   using GPUs 1–3.
2. Evaluate rung 9 on all nine source graphs using GPUs 0–3.

This yields 17 new cells: the Facebook column for rungs 1–8 and the complete rung-9
row. The all-nine merge uses the connected 119,228-node Facebook structural view.

## Tucker launch

Run from the isolated `prodigy-facebook` worktree:

```bash
tmux new-session -d -s nm-ladder-facebook \
  'export PATH="/home/mhchu/miniconda3/bin:$PATH"; \
   bash scripts/experiments/setup/nm_ladder_facebook/run_ladder_tucker.sh'
```

Use `DRY_RUN=1 bash .../run_ladder_tucker.sh` to validate the all-nine graph, the
eight historical checkpoints, and the planned GPU split without launching jobs.

Runtime model lists, logs, and status are written beneath the gitignored `run_logs/`
directory. Although current trainers also emit a terminal 50k checkpoint, all ladder
comparisons explicitly use `state_dict_40000.ckpt`.
