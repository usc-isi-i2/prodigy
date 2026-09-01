# MT transfer pilot

Early-budget test of the paper's supervised prompt-graph multi-task objective on
five social-graph classification datasets. It trains five MT specialists and five
paper-style NM+MT specialists, then evaluates the resulting two 5x5 matrices.

- MT uses real classification labels in few-shot prompt episodes.
- NM+MT alternates complete MT and NM episodes 1:1 through `MultiTaskSplitBatch`.
- Both arms receive 900 optimizer updates with batch size 8 and identical architecture.
- Evaluation uses 100 paired 3-shot episodes per model-target cell.
- The existing NM matrix is the comparison anchor and is not retrained.

The pilot is single-seed and early-budget. It is a screening experiment, not a final
publication estimate. Only GPUs 2 and 3 are permitted by the launchers.

Run in a dedicated Tucker worktree after a smoke test:

```bash
export PATH="/home/mhchu/miniconda3/bin:$PATH"
MODE=smoke SLOTS=2,3 bash scripts/experiments/setup/mt_transfer_pilot/run_train_tucker.sh
MODE=full bash scripts/experiments/setup/mt_transfer_pilot/run_train_tucker.sh
STATE_DIR="$PWD/state" bash scripts/experiments/setup/mt_transfer_pilot/make_model_list.sh
bash scripts/experiments/setup/mt_transfer_pilot/run_eval_tucker.sh
```
