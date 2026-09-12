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

Independent verification uses the existing `prodigy` environment (PyTorch 2.0.1),
with ordinary eager loading. It independently checks completeness of the nonzero
population, deterministic mini selection, every retained edge, edge attributes,
labels, targets, masks, user mapping and embedded payload against the original.

```bash
python scripts/graph_construction/verify_nonzero_feature_views.py \
  --root /dataMeR1/phil/data/graph_views/nonzero_features_v1 \
  --report /dataMeR1/phil/data/graph_views/nonzero_features_v1/production_verification.json
```

The catalog retains the originals' names and default evaluation selection. New
views have distinct keys ending `_nonzero_features_v1` and `default_eval: false`.
The two mini directory names end `_mini_500k`; seed 0 applies independently to
each graph. This does not create filtered mixtures or launch training.

The existing broad catalog test has four failures and two errors already present
at `d4f06374`, concerning Cora/PubMed and the public KG entries. They are unchanged
by this work. The focused nonzero-view preservation tests and catalog checks are
reported separately.

## Verified inventory

All eleven views passed independent production checks on 2026-09-12 UTC
with PyTorch `2.0.1+cu118`. Receipts are committed under `data/` beside these
construction scripts and retained beneath the graph-view root on Tucker.

All edge counts below are stored directed edges. Node counts include isolates.

| View | Nodes | Edges | Isolates |
|---|---:|---:|---:|
| `covid19_twitter` | 18,493,966 | 85,719,134 | 384,146 |
| `covid19_twitter_mini_500k` | 500,000 | 63,956 | 444,405 |
| `covid_political` | 78,672 | 180,928 | 24,930 |
| `cp_hk_twitter` | 256,798 | 837,495 | 28,952 |
| `election2020` | 78,932 | 2,818,603 | 0 |
| `facebook_page_reference` | 149,514 | 166,512 | 30,888 |
| `midterm` | 274,361 | 723,032 | 5,100 |
| `twibot20` | 144,274 | 1,769,115 | 2,041 |
| `ukr_rus_suspended` | 44,157 | 251,955 | 1,300 |
| `ukr_rus_twitter` | 7,973,536 | 57,964,016 | 166,401 |
| `ukr_rus_twitter_mini_500k` | 500,000 | 218,163 | 395,630 |

Uniform induced node sampling leaves 79.1% of Ukraine-mini and 88.9% of
COVID-mini isolated. The 500k cap equalizes stored node counts between these
two views, not their number of connected nodes or training edges.
