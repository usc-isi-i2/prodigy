# Interleaved neighbor MLP pairs (Election excluded)

28 unordered pairs among the eight gallery graphs other than Election 2020. Each model starts from seed-0 fresh weights and bias -log(5), uses one shared AdamW optimizer, and strictly alternates A/B minibatches 1:1. A is the earlier source in the catalog-derived experimental source order. No graph is removed when its individual validation plateaus.

Source-local sampling, architecture, and cached context match the bias gallery: 1536→256→256, ReLU, no dropout; fixed up-to-10 neighbor mean concatenated to node features; context edges disjoint from supervision; 1,024 positive edges per minibatch (partial last batches allowed), five uniform exact-nonedge negatives per positive. Negatives exclude all known edges and self-loops. AdamW lr 0.0005, weight decay 1e-5, gradient clip 1. FP32 with TF32 disabled.

Both graphs are validated every 2,000 total updates (1,000 per graph), using the original deterministic source-validation pairs. Checkpoint selection minimizes (BCE_A+BCE_B)/2. Stop after three checks without absolute mean improvement of 1e-4, counting stale checks after 2,500 total updates, matching the existing minimum-step convention. Safety cap: 100,000 TOTAL updates (50,000 per graph), flagged separately from convergence. Save per-source training losses, validation BCE/AUC, source update counts, best/latest/final checkpoints, optimizer and global RNG states. Sampler positions are not serialized, so exact interrupted-run replay is unsupported.

Compare against both sequential orders and singletons on the same eight targets (224 evaluation cells); unseen-source summaries use the six graphs outside each pair. Evaluation uses identical cached test pairs and freezes the decoder bias. Mean BCE is equally weighted by graph, not by graph size. This is convergence-selected training, not a matched-compute comparison with sequential runs; report total updates per method and flag cap hits.

Launch in a dedicated Tucker worktree after read-only `plan` preflight:

```bash
export PATH="/home/mhchu/miniconda3/bin:$PATH"
bash scripts/run_interleaved_mlp_pairs.sh /dataMeR1/phil/gfm/mixture-scaling/state/interleaved_mlp_pairs_noelec_s0
```

Four workers share a locked queue on GPUs 0–3. Existing partial runs are refused. Evaluation starts only after all training workers succeed. Aggregate receipt requires all 224 cells. The `plan` phase checks graph, context, split, and evaluation-pair artifacts before launch.
