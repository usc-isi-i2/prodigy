# Strong raw-feature MLP

This experiment replaces the earlier 100-update MLP sanity check with a tuned,
convergence-trained supervised baseline.  It deliberately uses only the graph
artifact's raw node features (no topology), the repository's deterministic
60/20/20 stratified node split, and validation-only model selection.

The default grid reports balanced label budgets of 10, 100, 500, and 1,000 per
class plus the complete training split.  For each target and budget, seed 0
selects an architecture/optimizer configuration by validation ROC-AUC.  That
configuration is then locked and trained with five seeds.  Test labels are read
only after the best validation checkpoint has been selected.

Canonical artifacts are written below the supplied output directory:

- `selection.jsonl`: every validation-only tuning candidate;
- `results.jsonl`: one final test result per target, budget, and seed;
- `curves/*.jsonl`: epoch-level train/validation histories;
- `checkpoints/*.pt`: best model plus optimizer state and effective config;
- `protocol.json`: the complete experiment contract.

W&B runs are offline and supplementary. JSONL, checkpoints, and protocol files
remain canonical.

On Tucker, from an isolated worktree:

```bash
tmux new-session -d -s strong_mlp \
  'export PATH="/home/mhchu/miniconda3/bin:$PATH"; bash scripts/experiments/setup/strong_mlp/run_tucker.sh'
```

Use `DRY_RUN=1` to print the command without executing it.  The run is light
enough for one owned GPU and refuses devices outside 0--3.
