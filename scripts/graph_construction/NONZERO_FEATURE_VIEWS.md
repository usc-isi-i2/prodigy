# Nonzero-feature graph views v1

Reusable induced graph objects live under
`/dataMeR1/phil/data/graph_views/nonzero_features_v1/<dataset_key>/graph.pt`.
The original source graphs are read-only. Exact zero rows are removed; nonzero
isolates and arbitrarily small nonzero vectors remain. Nonfinite features fail
explicitly. Features, labels, node targets, masks, user IDs, edge attributes,
all existing edge views and embedded PyG payloads are filtered consistently.
`original_node_ids` maps each view row to its original source row. Existing
construction statistics are retained under `source_graph_metadata` so stale
counts cannot masquerade as view statistics.

Ukraine and COVID additionally have `_mini_500k` variants: uniform sampling
without replacement from the nonzero population, seed 0, restored source order.
Every original edge between selected nodes is retained, including held-out and
temporal edges in their existing separate views. Sampling never removes more
edges after inducing. LP consumers must reject negatives against all known
edges, including held-out targets; this builder does not change samplers.

Use an isolated worktree and tmux on Tucker. The CPU-only mapped loading
benchmark environment is used for construction; production environments remain
unchanged. Output is standard PyTorch graph serialization. Loading with the
production environment is checked separately before publication.

```bash
/dataMeR1/phil/conda_envs/prodigy-loadbench/bin/python -m unittest discover \
  -s scripts/graph_construction/tests -p test_nonzero_feature_views.py
/dataMeR1/phil/conda_envs/prodigy-loadbench/bin/python \
  scripts/graph_construction/nonzero_feature_views.py \
  --output /dataMeR1/phil/data/graph_views/nonzero_features_v1
```

Each destination must be new. A sidecar `metadata.json` is written only after
source-order verification of all induced edge views, exact feature checks,
full serialization round-trip checks, and unchanged source size/mtime checks.
Source identity is path, byte size and nanosecond modification time, not a
cryptographic content hash. Output counts include retained isolates.
