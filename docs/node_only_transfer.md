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
  The feature matrix is copied to the assigned GPU once and endpoint rows are gathered
  there; no neighborhoods are sampled or collated, and only compact endpoint indices
  cross the host-device boundary per update. Uniform random
  training negatives exclude self-loops, matching the earlier loader's approximate
  negative-sampling contract. Evaluation negatives are unchanged.

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

Only GPUs 0–3 are accepted. Launchers refuse occupied GPUs, skip completed
models, and reject ambiguous partial run directories. Canonical aggregate outputs
are `results/node_only_transfer/aggregated/node_mlp_{fp,lp}_transfer.tsv`.

The GPU-resident change keeps the canonical 1,024-positive batch size and therefore
does not alter the optimizer-update protocol. On `covid19_twitter`, a fixed-example
benchmark measured 882 versus 105 updates/second for GPU-resident versus CPU-direct
batches (8.4x), with identical endpoint pairs and final loss. The 65.8 GiB feature
matrix fits on an 80 GB H100; measured peak allocation was 66.0 GiB at this batch size.
