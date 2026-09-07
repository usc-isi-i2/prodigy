# Frozen message-content follow-up

The completed role-topology replay found strong deletion effects but weak
degree-preserving rewiring effects. A direct operator probe found that these
checkpoints' SAGE inference runtime reports `aggr="mean"` while executing a
cached PyG `SumAggregation` module. This experiment tests the implications;
it does not patch production code, assert a training-time defect, or retrain.

Reuse the same six fixed HK/Ukraine controls, all five targets and both cached
streams. Sixteen conditions per model/target/stream give 960 cells: baseline
plus each of five message changes on query only, support only and both roles.

- Actual mean: use true arithmetic mean instead of the observed sum.
- No message bias: retain `W x` but subtract the per-message affine term `b`.
- Bias only: retain `b` but remove the feature term `W x` from each message.
- Mean message: every sender emits its subgraph's mean projected real-node
  feature; preserve degrees and node-specific self projections. This retains
  aggregate feature content but removes sender identity within the subgraph.
- Zero messages: must reproduce existing edge-removal metrics for each role.

This is a follow-up informed by earlier results, not an untouched confirmation.
Primary question: do actual mean aggregation or affine-bias removal reproduce
the political/page support-removal improvements, and at what cost elsewhere?
Bias-only and mean-message conditions distinguish feature-specific message
content from degree-dependent effects. No candidate is selected using queries;
report the complete panel and keep the original trained weights fixed.

All baseline and zero-message AUC/accuracy/F1/NLL values must match the prior
replay to 1e-6. Direct first-batch role-local changes must match factored
predictions; untouched-role encodings must be bit-exact. Checkpoint tensors,
inputs and the original aggregation operator must be restored after changes.
The mean/sum primitive is tested on every loaded checkpoint, not inferred from
a configuration string. The three initialization seeds remain the only
training replications.

## Run on Tucker

After Git sync in an idle isolated worktree:

```bash
bash scripts/experiments/setup/target_performance_mechanisms/run_message_content_tucker.sh --output log/message_content_full_20260907 --dry-run
bash scripts/experiments/setup/target_performance_mechanisms/run_message_content_tucker.sh --output log/message_content_smoke_20260907 --targets covid_political --seeds 0 --max-batches 2
tmux new-session -d -s mechanism-message-content 'cd /dataMeR1/phil/gfm/prodigy-role-topology && export PATH="/home/mhchu/miniconda3/bin:$PATH" && bash scripts/experiments/setup/target_performance_mechanisms/run_message_content_tucker.sh --output log/message_content_full_20260907 > log/message_content_full_20260907.log 2>&1'
```

The launcher hides all GPUs and bounds computation to four CPU threads.
Outputs are new directories, with complete per-query logits on Tucker and
compact metrics/receipts written after each target/stream. A smoke run is never
reported as a completed experimental panel.
