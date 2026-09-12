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

## Edge-sampled 500k minis

The additional directories `ukr_rus_twitter_mini_500k_edge_sampled_s0` and
`covid19_twitter_mini_500k_edge_sampled_s0` select endpoints from uniform random
stored edge-row draws, with replacement, seed 0. Self-loop draws are skipped.
An edge introducing too many nodes for the exact 500,000-node cap is skipped
whole; endpoints are never trimmed. After selection, **all** parent edges
between selected nodes are retained, including unsampled edges and every
existing static/temporal edge view. Original graph splits are preserved.

The parent is the already verified full nonzero graph, loaded with mmap. Only
selected features are gathered; full nonzero features are not scanned again.
Two graph builds can run concurrently with four CPU tensor threads apiece.
This selection favors high-degree nodes and does not promise one connected
component. Every selected node has a sampled non-self connection in the full
mini. The static training-background view can still contain isolates after
held-out edges are excluded; that count is recorded separately.

`parent_node_ids` maps to the nonzero parent; `original_node_ids` maps through
that parent to the original unfiltered graph. `selection.pt` records accepted
parent edge IDs as witnesses for the selected endpoints. Source identities,
all induced views and aligned attributes, and full mini serialization are
checked before writing `metadata.json`. The earlier `_mini_500k` directories
are the uniform-node variants and are retained unchanged.

```bash
/dataMeR1/phil/conda_envs/prodigy-loadbench/bin/python \
  scripts/graph_construction/edge_sampled_minis.py --dataset ukr_rus_twitter
/dataMeR1/phil/conda_envs/prodigy-loadbench/bin/python \
  scripts/graph_construction/edge_sampled_minis.py --dataset covid19_twitter
```

Verified results (stored directed edges):

| Edge-sampled mini | Nodes | Full edges | Full isolates | Static-background isolates | Build + verification |
|---|---:|---:|---:|---:|---:|
| Ukraine | 500,000 | 25,710,237 | 0 | 7,817 (1.56%) | 23.7 s |
| COVID | 500,000 | 14,072,647 | 0 | 10,800 (2.16%) | 33.7 s |

Both builds ran concurrently and passed production PyTorch 2.0.1 loading,
feature/mapping checks and recomputed inventory counts. Times include parent
loading, sampling, induction, full edge/attribute checks and mini round-trip,
excluding code preparation and the separate production loading checks. Filesystem
caches were warm. Receipts are in `data/edge_sampled_minis_*.json`.

These minis contain zero isolates in the **full** graph; existing static holdout
edges are still excluded from the training background. High-degree selection
bias is substantial: Ukraine retains 44.4% of parent edges with 6.3% of its nodes;
COVID retains 16.4% of parent edges with 2.7% of its nodes.
