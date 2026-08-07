# Path length versus node-feature distance

This analysis asks a within-graph question that the graph-divergence sweep does
not: **as two nodes get farther apart in the graph, do their feature vectors also
get farther apart?** It complements the existing between-graph feature-cloud and
topology divergences.

The primary sample is made of matched anchor blocks. Each retained anchor has one
endpoint at exact undirected shortest-path distance 1, 2, and 3, plus one endpoint
verified to be farther than 3 hops or disconnected. All five nodes must have
nonzero text features. The undirected view matches the default PRODIGY message
passing; the 1–3-hop range covers the one- and two-hop regimes used in the current
experiments without attempting infeasible all-pairs shortest paths on the 23M-node
COVID graph.

The output includes three kinds of evidence:

1. Mean, spread, paired 1-hop-to-far effect, and descriptive Spearman trend for
   cosine and Euclidean feature distance.
2. A held-out adjacent-versus-far classifier built from symmetric pair features
   (`abs(x_u-x_v)`, pair mean, and elementwise product). A single-coordinate probe
   and an L1-sparse probe specifically test whether one or a few informative GTE
   dimensions are drowned out by whole-vector distances. Train/test pairs are
   node-disjoint by deterministic node hash: both endpoints must fall entirely in
   the training or test partition.
3. Mean random-pair cosine and Euclidean distances within and between graphs in the
   same units, unlike centroid cosine or proxy-A-distance.

## Tucker run

Use the `prodigy` environment. The defaults cover the eight single-source retweet
graphs in `docs/graph_catalog.json` and retain 20,000 complete anchor blocks per
graph:

```bash
export LD_LIBRARY_PATH="/home/mhchu/miniconda3/envs/prodigy/lib:${LD_LIBRARY_PATH:-}"
/home/mhchu/miniconda3/envs/prodigy/bin/python \
  scripts/experiments/analysis/path_feature_coupling/analyze_path_feature_coupling.py
```

Pilot one smaller graph first:

```bash
/home/mhchu/miniconda3/envs/prodigy/bin/python \
  scripts/experiments/analysis/path_feature_coupling/analyze_path_feature_coupling.py \
  --graphs midterm --blocks 1000 \
  --out /tmp/path_feature_coupling_midterm_pilot.json
```

## Interpretation cautions

- The short-distance endpoints are random-walk sampled, not uniform over every
  possible node pair at a given distance. Matching all distances to the same
  anchor removes a major node-marginal confound, but results describe this sampling
  regime.
- `>3_or_disconnected` is an ordinal comparison bucket, not a known numeric path
  length. The reported 1–3 correlation uses exact distances only; the second
  correlation labels the far bucket as 4 and is explicitly descriptive.
- Conditioning on nonzero text features isolates semantic-feature geometry. Missing
  text itself may be informative to a GNN and should be analyzed separately if it
  becomes part of the mechanism claim.
- A strong sparse probe with a small cosine trend means “predictive feature
  direction hidden by aggregation,” not necessarily that the pretrained GNN uses
  that direction. Establishing use requires an intervention that masks or permutes
  the selected coordinates and re-evaluates the encoder.
