# Node-only MLP transfer

This experiment isolates transfer carried by raw node features without learned
message passing. It trains one MLP specialist on each of the nine Social-9
sources under two self-supervised objectives and evaluates every specialist on
every eligible target.

- **FP:** coordinate-masked feature prediction. Fifty percent of each node's
  768 feature coordinates are replaced by a learned mask token. The MLP predicts
  the masked coordinates from the same node's visible coordinates. Full-node
  masking is intentionally not used because it would leave a node-only model
  with no signal.
- **LP:** source-confined link prediction. Each endpoint is encoded independently
  from its raw feature vector; endpoint dot products train the model. Evaluation
  uses the existing deterministic background/holdout partitions, degree-matched
  negatives, validation-locked cosine orientation, and leakage/sensitivity gates.

Checkpoint selection uses source SSL validation only. Target performance never
selects a checkpoint. FP evaluates all nine targets with ten deterministic mask
replicates; static LP evaluates the six targets covered by the canonical repaired
benchmark. The final long-form tables therefore contain 81 FP cells and 54 LP cells.

Run from an isolated Tucker worktree after activating the `prodigy` environment:

```bash
bash scripts/run_node_only_transfer_tucker.sh fp gate
bash scripts/run_node_only_transfer_tucker.sh lp gate
bash scripts/run_node_only_transfer_tucker.sh fp full
bash scripts/run_node_only_transfer_tucker.sh lp full
bash scripts/eval_node_only_transfer_tucker.sh fp
bash scripts/eval_node_only_transfer_tucker.sh lp
```

Only GPUs 2 and 3 are accepted. Launchers refuse occupied GPUs, skip completed
models, and reject ambiguous partial run directories. Canonical aggregate outputs
are `results/node_only_transfer/aggregated/node_mlp_{fp,lp}_transfer.tsv`.
