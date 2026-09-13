# Complete canonical NM inputs

Reconstruct the exact 512 held-out test episodes per target from the canonical
saved plans and RNG reset, without any model forwards. The reused TrainerFS
initializer does load the configured Ukraine checkpoint on CPU while constructing
the loaders; no prediction or learned embedding is computed from it.
Run on Tucker in the `prodigy` environment. Runtime code is isolated from current
model development on branch `codex/nm-complete-input-audit-20260908`, revision
`19a9a6a8`, derived from the successful support-resampling runtime.

```bash
export PATH="/home/mhchu/miniconda3/bin:$PATH"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate prodigy
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
python -u scripts/experiments/setup/nm_complete_input_audit/run.py \
  --threads 4 --out /dataMeR1/phil/gfm/error_audit/nm_complete_inputs_20260908_v2
```

The output directory must not exist. Use a dedicated worktree and tmux session.
`--audit-root` and `--bio-root` override the canonical private input roots.

Each target saves 16 private batch files, covering 32 episodes apiece. A batch
preserves the complete PyG graph (all sampled global IDs, edges, edge attributes,
graph boundaries, centers, pooling nodes and mappings) and every remaining model
input tensor. Only `graph.x` is replaced by a zero-row placeholder; reconstruct
it using the `restore` helper and the specified backing graph's feature matrix.
The synthetic pooling node has ID −1 and receives a zero feature vector.

Each file is immediately reloaded, its features restored, and its full tensor
hash compared to the original in-memory batch. The concatenated batch hash must
also match the earlier canonical audit for each target. Field order is preserved
because the canonical hashing helper walks PyG attributes in insertion order.
The receipt includes the feature artifact path/shape, all compact-file hashes,
and expected/original/reloaded full-input hashes. Changing the feature artifact
requires rechecking these hashes; an ID-only archive is not self-contained.

`query_geometry_private.csv` retains per-query descriptors of exact sampled
inputs, joined to both models' original predictions. All 30 candidate support
classes are compared. Descriptors include node-set Jaccard against each class's
support union and cosine similarities between normalized query summaries and
the average of three normalized support summaries. Summaries are center vector,
mean real-node feature vector, context-only mean, and concatenated normalized
center/context means. These are diagnostic summaries, not learned embeddings
or the actual architecture's flattened input. Means ignore the synthetic node;
context-only means additionally exclude the center. Invalid zero-vector
comparisons are tracked. No model decision is regenerated.

The local analysis helper is
`scripts/experiments/analysis/evaluation/error_audit/summarize_nm_complete_inputs.py`.
It reads `receipt.json` and each target's private geometry CSV and emits only
aggregate statistics; keep private batch files and node-level tables out of git.
