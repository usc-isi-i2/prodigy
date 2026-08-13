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
   plus 1/4/16/64-coordinate and L1-regularized probes specifically test whether
   one or a few informative GTE dimensions are drowned out by whole-vector distances.
   The node-dimension probes include that coordinate's absolute difference, pair
   mean, and product, so they can detect both similarity and localized feature
   regions. Train/test pairs are
   node-disjoint by deterministic node hash: both endpoints must fall entirely in
   the training or test partition.
3. Mean random-pair cosine and Euclidean distances within and between graphs in the
   same units, unlike centroid cosine or proxy-A-distance.

A second, uniform-pair diagnostic answers the follow-up about distances beyond
three hops and exports every original feature coordinate. It samples nodes exactly
as independent uniform center choices, computes exact finite shortest-path lengths,
and records disconnected pairs separately. For each coordinate it correlates path
length with three symmetric pair summaries: absolute difference, pair mean, and
elementwise product. It also reports a node-disjoint edge-versus-uniform AUC. A
separate held-out univariate Gaussian probe measures how well each raw coordinate
predicts graph identity, both across all eight graphs and across the six graphs made
with the same feature pipeline.

The complete outputs are:

- `data/dimension_diagnostics.json`: full sampling metadata and nested results.
- `data/node_distance_per_dimension.csv`: one row per graph and feature coordinate.
- `data/graph_identity_per_dimension.csv`: one row per scope and feature coordinate.
- `data/neighbor_augmented_features.json`: matched raw-versus-neighborhood distance
  metrics and graph-identity probes for the training-style neighborhood summary.
- `data/neighbor_augmented_3d.csv`: held-out 3D PCA and graph-label-supervised LDA
  coordinates for raw centers, sampled-neighbor means, and their concatenation.
- `figures/neighbor-augmented-feature-space.html`: portable interactive 3D view
  with PCA/LDA and raw-center/neighbor-mean/concatenated-space selection.

## Tucker run

Use the `prodigy` environment. The defaults cover the eight single-source retweet
graphs in `docs/graph_catalog.json` and retain 20,000 complete anchor blocks per
graph:

```bash
export LD_LIBRARY_PATH="/home/mhchu/miniconda3/envs/prodigy/lib:${LD_LIBRARY_PATH:-}"
/home/mhchu/miniconda3/envs/prodigy/bin/python \
  scripts/experiments/analysis/graph_characterization/structure_feature_coupling/path_feature_coupling/analyze_path_feature_coupling.py
```

Pilot one smaller graph first:

```bash
/home/mhchu/miniconda3/envs/prodigy/bin/python \
  scripts/experiments/analysis/graph_characterization/structure_feature_coupling/path_feature_coupling/analyze_path_feature_coupling.py \
  --graphs midterm --blocks 1000 \
  --out /tmp/path_feature_coupling_midterm_pilot.json
```

Run the uniform-pair and per-dimension diagnostic, then export flat tables:

```bash
/home/mhchu/miniconda3/envs/prodigy/bin/python \
  scripts/experiments/analysis/graph_characterization/structure_feature_coupling/path_feature_coupling/analyze_dimension_diagnostics.py
/home/mhchu/miniconda3/envs/prodigy/bin/python \
  scripts/experiments/analysis/graph_characterization/structure_feature_coupling/path_feature_coupling/export_dimension_tables.py
```

Compare raw centers with the center-plus-neighbor-mean representation. The default
samples up to 100 undirected neighbors without replacement, matching the historical
one-hop sampler:

```bash
/home/mhchu/miniconda3/envs/prodigy/bin/python \
  scripts/experiments/analysis/graph_characterization/structure_feature_coupling/path_feature_coupling/analyze_neighbor_augmented_features.py
```

## Interpretation cautions

- The short-distance endpoints are random-walk sampled, not uniform over every
  possible node pair at a given distance. Matching all distances to the same
  anchor removes a major node-marginal confound, but results describe this sampling
  regime.
- `>3_or_disconnected` is an ordinal comparison bucket, not a known numeric path
  length. The reported 1–3 correlation uses exact distances only; the second
  correlation labels the far bucket as 4 and is explicitly descriptive.
- The uniform-pair diagnostic resolves that bucket: connected uniform pairs have
  exact finite lengths, while disconnected pairs have no finite path length and are
  summarized separately rather than assigned an arbitrary large number.
- Pair mean and product can reveal that edge endpoints occupy a special feature
  region, including degree or sampling effects. They are GNN-accessible structure
  signals, but are not by themselves evidence of feature homophily. Absolute
  difference is the direct coordinate-wise smoothness statistic.
- Raw embedding axes depend on the embedding pipeline. The same-pipeline graph-
  identity scope is the cleaner comparison; the all-graph scope also contains the
  known political-graph pipeline shift.
- The concatenation is an information-level diagnostic, not the literal hidden state
  of the trained encoder. The default SAGE layer projects and mean-aggregates neighbor
  messages, applies its neighbor MLP, then adds a separately projected center (and an
  optional residual) before normalization/ReLU. The raw concatenation asks what is
  jointly available before those learned transformations.
- The LDA view consists of learned linear combinations of all input coordinates, not
  three selected raw dimensions. Its axes are fit with graph labels on a balanced 70%
  training split, and the interactive plot shows only the held-out 30%. PCA is fit on
  the same training nodes without labels so the two views use identical held-out data.
- Conditioning on nonzero text features isolates semantic-feature geometry. Missing
  text itself may be informative to a GNN and should be analyzed separately if it
  becomes part of the mechanism claim.
- A strong sparse probe with a small cosine trend means “predictive feature
  direction hidden by aggregation,” not necessarily that the pretrained GNN uses
  that direction. Establishing use requires an intervention that masks or permutes
  the selected coordinates and re-evaluates the encoder.
