# Graph divergence: pairwise comparison of all source social graphs

Diagnoses **how the source graphs differ**, to ground graph-transfer analysis.
Instead of one blended similarity score, it separates three axes and then measures
the coupling a message-passing GNN actually transfers:

1. **Topology** — degree distributions (directed in/out), density, reciprocity,
   degree assortativity, approximate clustering, largest WCC/SCC fraction.
2. **Features** — GTE text embeddings (`x`: user bios for Twitter/X, page
   descriptions for Facebook; zero-filled when text is missing): missing-text
   rate, feature norm, effective dimensionality.
3. **Feature–structure coupling** — edge feature homophily vs. a random-pair
   baseline, Dirichlet energy, and label mixing where labels exist. Label output
   includes raw same-label edge homophily, the directed endpoint-marginal chance
   baseline, Newman's nominal assortativity, per-class conditional rates, and the
   effective labeled-edge count.

For every **ordered pair** of graphs it also computes: in/out degree-distribution
KS distance, feature centroid cosine distance, Frechet distance, RBF-MMD², and
proxy-A-distance (a logistic domain classifier's separability of the two feature
clouds; 0 = indistinguishable, 2 = perfectly separable).

## Files

- `compute_graph_divergence.py` — the runner. Loads each graph, computes all of the
  above, writes one JSON artifact.
- Plots + write-up: [`scripts/experiments/analysis/graph_divergence/graph_divergence.ipynb`](../../plotting/graph_divergence/graph_divergence.ipynb),
  reading `graph_divergence_data.json` in that folder. Running the notebook exports
  the scalar `figures/per_graph_summary.csv` and long-form class-conditional
  `figures/per_class_label_mixing.csv` findings tables.

## Graphs compared

Single-source social graphs only (merged graphs are unions of these, not
independent domains). Defaults, relative to `--data-root` (`/dataMeR1/phil/data`).
Facebook uses the 119,228-node structural view used by the nine-graph transfer
experiments, excluding the 30,772 deterministically selected attributed isolates:

| name | path | ~nodes | ~edges | labels |
|------|------|-------:|-------:|--------|
| covid19_twitter | `covid19_twitter/graphs/retweet_graph_parquet.pt` | 23.0M | 107.2M | — |
| ukr_rus_twitter | `ukr_rus_twitter/graphs/retweet_graph_parquet.pt` | 10.4M | 76.9M | — |
| midterm | `midterm/graphs/retweet_graph_parquet.pt` | 342k | 900k | — |
| cp_hk_twitter | `cp_hk_twitter/graphs/retweet_graph.pt` | 334k | 1.18M | — |
| twibot20 | `twibot20/graphs/retweet_graph.pt` | 163k | 2.01M | human/bot |
| election2020 | `election2020/graphs/retweet_graph.pt` | 79k | 2.82M | conservative |
| covid_political | `covid_political/graphs/retweet_graph.pt` | 79k | 181k | conservative |
| ukr_rus_suspended | `ukr_rus_suspended/graphs/retweet_graph.pt` | 72k | 354k | suspended |
| facebook_page_reference | `facebook_page_reference/graphs/page_reference_structural.pt` | 119k | 168k | page category, admin country, verified, regression targets |

## How to run (on Tucker)

Loading a 75GB graph is heavy, but the runner memory-maps the feature tensor
(`torch.load(..., mmap=True)`) and subsamples nodes/edges for every feature metric,
so only `edge_index` is materialised in full — peak RAM is a few GB and the whole
9-graph sweep takes ~7 min. Non-interactive shells don't have `conda` on PATH, so
call the env's Python directly:

```bash
export LD_LIBRARY_PATH="/home/mhchu/miniconda3/envs/prodigy/lib:${LD_LIBRARY_PATH:-}"
/home/mhchu/miniconda3/envs/prodigy/bin/python \
    scripts/experiments/analysis/graph_divergence/compute_graph_divergence.py \
    --data-root /dataMeR1/phil/data \
    --out scripts/experiments/analysis/graph_divergence/data/graph_divergence_data.json
```

Then commit `graph_divergence_data.json` (≈160 KB) and pull it to the laptop; the
notebook reads it directly. Useful flags: `--graphs a,b,c` (subset),
`--feat-sample`, `--edge-sample`, `--max-edges-exact`, `--seed`.

## Data artifact schema (`graph_divergence_data.json`)

```
meta:        generated_at, data_root, git_commit, hostname, seed, config{...},
             graph_paths{name -> path relative to data_root}
graphs:      [ordered graph names]
per_graph:   name -> { topology scalars, in/out degree CCDFs, feature scalars,
                       coupling scalars, class_balance, label_homophily,
                       label_homophily_expected, label_assortativity_newman,
                       labeled_edge_count, label_mixing{classes, counts,
                       endpoint counts, per-class same-label rates}, ... }
pairwise:    metric -> NxN matrix (row/col order == `graphs`), for metrics
             indegree_ks, outdegree_ks, feat_centroid_cosdist, feat_frechet,
             feat_mmd2, proxy_a_distance
```

## Method notes / caveats

- **Feature stats are on nodes with nonzero text embeddings.** The historical JSON
  key `missing_bio_rate` is retained for schema compatibility; for Facebook it
  means missing page descriptions. It is estimated from a uniform node sample and
  is a first-order confound: a graph with many empty texts has less feature signal.
- **Cross-pipeline/platform confounds.** `election2020` / `covid_political` embeddings use
  meanpool pooling with 0% missing bios, unlike the other graphs (zero-filled). A
  large feature divergence to those two is partly construction, not domain shift.
  Facebook is a page-reference graph with page-description features rather than a
  Twitter/X retweet graph with user-bio features, so its divergence also combines
  platform, relation, and population differences.
- **Subsampling** (seeded): `--feat-sample` non-missing nodes per graph for feature
  clouds; `--edge-sample` edges for homophily/coupling; degree KS on ≤50k-degree
  samples; MMD on ≤`--mmd-cap` points/graph; clustering on `--clustering-nodes`
  sampled nodes with a per-hub neighbor cap (so `avg_clustering_approx` is an
  estimate). Reciprocity/assortativity are exact unless E > `--max-edges-exact`.
- **Label imbalance and direction are handled explicitly.** Raw label homophily is
  descriptive only. `label_homophily_expected` is computed from the source and
  destination label marginals among sampled labeled edges—not from the global node
  class balance—and `label_assortativity_newman` chance-corrects the raw rate using
  that baseline. Inspect `labeled_edge_count`, `labeled_edge_fraction`, and the
  per-class rates before comparing graphs with different label coverage.
- Every metric is wrapped so one failure yields `null` rather than aborting the run.
