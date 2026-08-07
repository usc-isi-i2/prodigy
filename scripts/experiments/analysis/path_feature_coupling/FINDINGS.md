# Findings: path length versus node-feature distance

*Full eight-retweet-graph sweep, 20,000 matched anchor blocks per graph, seed 0.
Generated on Tucker 2026-08-07. Method and reproduction command are in
[`README.md`](README.md); raw results are in
[`data/path_feature_coupling.json`](data/path_feature_coupling.json).*

## Takeaway

**Whole-vector feature distance increases only weakly with exact local path
length, but that average hides predictive low-dimensional feature structure.**
Across exact undirected distances 1, 2, and 3, the within-graph Spearman
correlation between path length and cosine distance is only **0.007–0.139**
(mean **0.063**). Seven of eight graphs nevertheless show a clearer adjacent-to-
far/disconnected shift: mean cosine distance rises by 0.020–0.056, a standardized
effect of 0.30–0.70. Ukraine-suspended is the exception (effect −0.04).

The user's sparse-dimension concern is borne out. On node-disjoint held-out pairs,
one train-selected original embedding dimension predicts adjacent versus
far/disconnected at AUC **0.56–0.67**, and four dimensions reach **0.61–0.76** on
the seven graphs with a scalar gap. The same four dimensions used only as absolute
coordinate differences are weaker (AUC 0.55–0.64): much of the recoverable signal
is in a feature region or interaction, not ordinary feature smoothness. A whole-
vector cosine average can therefore look nearly flat even when a GNN could exploit
a small subspace.

This is evidence of *available* feature–structure coupling, not yet proof that the
trained GNN uses those coordinates. The causal follow-up is to mask or permute the
selected dimensions and re-evaluate the frozen encoder.

## What uniform center sampling actually produces

The follow-up sampled independent nodes uniformly, matching the way training can
choose distant centers, and computed exact undirected shortest paths rather than
stopping at three hops. **Connected uniform pairs are not hundreds of hops apart in
these graphs.** Their median distance is 3–5, their 99th percentile is 4–10, and the
largest distance observed in this sweep is 14. The genuinely far case is often
disconnection: its probability ranges from 0.0% to 51.8% across graphs.

| graph | disconnected | finite median | finite mean | p90 | p99 | sampled max |
|---|---:|---:|---:|---:|---:|---:|
| covid | 3.0% | 5 | 4.75 | 6 | 7 | 10 |
| ukraine | 1.8% | 4 | 4.37 | 5 | 6 | 11 |
| midterm | 19.2% | 5 | 5.32 | 8 | 10 | 14 |
| hongkong | 21.0% | 4 | 3.55 | 5 | 6 | 8 |
| twibot20 | 1.5% | 4 | 4.05 | 5 | 6 | 11 |
| election2020-political | 0.0% | 3 | 2.57 | 3 | 4 | 5 |
| covid-political | 51.8% | 4 | 4.12 | 6 | 7 | 11 |
| ukraine-suspended | 42.7% | 4 | 3.86 | 5 | 7 | 10 |

This changes the interpretation of the earlier `>3_or_disconnected` bucket. It is
a good operational contrast to adjacency, but it mixes moderately distant connected
pairs with pairs in different components. The exact finite-distance analysis keeps
those cases separate.

## Every feature dimension versus node distance

Yes: all 768 dimensions are now tested separately. Because node distance belongs to
an unordered pair, each raw coordinate is represented by three symmetric quantities:
`|x_u-x_v|` (coordinate-wise smoothness), `(x_u+x_v)/2` (feature region), and
`x_u x_v` (same-sign/magnitude interaction). Across the eight graphs, the strongest
single-coordinate Pearson correlation with exact finite path length is only
**0.200–0.347 in absolute value**. This agrees with the weak whole-vector trend.

But a more training-relevant contrast is much stronger. A single coordinate can
distinguish a random edge from a uniformly sampled node pair with held-out,
node-disjoint AUC **0.701–0.908** across the eight graphs. The best values are 0.908
for COVID (dimension 343, pair mean), 0.880 for Ukraine (dimension 126, product),
0.800 for Hong Kong (dimension 178, mean), and 0.765 for Ukraine-suspended
(dimension 595, mean). Thus the user's concern is correct: strong localized
structure signals can be drowned out by cosine or Euclidean aggregation over 768
coordinates.

Most winners use pair mean or product rather than absolute difference. That still
gives the GNN useful information about which feature regions tend to participate in
edges, but it should not be described as pure homophily or smoothness.

## Every raw dimension versus graph identity

All 768 raw coordinates were also evaluated as one-dimensional graph classifiers on
held-out nodes. Across all eight graphs, the best coordinate reaches **18.8%**
balanced accuracy (12.5% chance), and the strongest coordinate/graph one-vs-rest AUC
is **0.746**. Restricting to the six same-pipeline graphs gives **22.7%** balanced
accuracy (16.7% chance) and a maximum one-vs-rest AUC of **0.668**. Individual axes
therefore carry real graph-domain information, but no single raw coordinate nearly
determines graph identity. The much stronger previously observed full-feature
graph-domain separation is multivariate.

No coordinate was singled out in advance: the analysis ranks all 768. Across all
eight graphs the best univariate graph-ID coordinates are 194, 454, 754, 119, 670,
656, 712, and 320 (balanced accuracy 18.3–18.8%). In the cleaner same-pipeline scope
the leaders are 179, 670, 320, 147, 414, 767, 295, and 543 (21.8–22.7%). These
indices are empirical ranked signals, not semantically stable named concepts.

The exhaustive tables are
[`node_distance_per_dimension.csv`](data/node_distance_per_dimension.csv) and
[`graph_identity_per_dimension.csv`](data/graph_identity_per_dimension.csv); the
sampling metadata and full nested results are in
[`dimension_diagnostics.json`](data/dimension_diagnostics.json).

## Results

All probes distinguish exact distance 1 from a uniformly sampled endpoint verified
to be farther than 3 hops or disconnected. A node dimension contributes its
absolute pair difference, pair mean, and elementwise product. Dimension selection
uses training nodes only; test pairs contain only nodes absent from training.

| graph | cosine ρ, exact 1–3 | mean cosine, d=1 → far | standardized gap | 1 node dim AUC | 4 node dims AUC | full pair-probe AUC |
|---|---:|---:|---:|---:|---:|---:|
| covid | 0.080 | 0.462 → 0.503 | 0.580 | 0.595 | 0.636 | 0.773 |
| ukraine | 0.084 | 0.468 → 0.504 | 0.496 | 0.664 | 0.698 | 0.803 |
| midterm | 0.024 | 0.428 → 0.468 | 0.503 | 0.601 | 0.627 | 0.750 |
| hongkong | 0.079 | 0.472 → 0.504 | 0.472 | 0.667 | 0.764 | 0.822 |
| twibot20 | 0.139 | 0.439 → 0.495 | 0.701 | 0.559 | 0.614 | 0.761 |
| election2020-political | 0.032 | 0.402 → 0.429 | 0.355 | 0.589 | 0.673 | 0.771 |
| covid-political | 0.062 | 0.440 → 0.462 | 0.302 | 0.635 | 0.713 | 0.796 |
| ukraine-suspended | 0.007 | 0.480 → 0.476 | −0.038 | 0.590 | 0.582 | 0.549 |

The local 1→2→3 changes are generally positive but not strictly monotone, and are
very small relative to pair-to-pair dispersion. Treating the far bucket as ordinal
bucket 4 raises the descriptive rank correlation to 0.110–0.248 on those seven graphs, but it remains
only 0.019 for ukraine-suspended.

## Relation to the between-graph result

The same averaging problem appears between graphs. Directly sampled random-pair
cosine distance averages **0.474 within graphs** and **0.489 between graphs**, with
heavily overlapping ranges (within 0.407–0.501; between 0.431–0.508). Across the 28
graph pairs, this mean pairwise cosine distance is not positively aligned with the
much more sensitive graph-domain proxy-A-distance (descriptive Spearman ρ = −0.175),
even though proxy-A-distance ranges from 0.19 to 1.79.

So the combined read is:

1. **Within a graph:** exact path length has weak correlation with whole-vector
   cosine/Euclidean distance.
2. **Between graphs:** average whole-vector distances also compress important
   distribution differences; a learned domain separator sees much more.
3. **Within and between graphs:** low-dimensional or multivariate discriminative
   directions can coexist with small average distances. Scalar cosine/Euclidean
   summaries are useful diagnostics, but cannot rule out GNN-usable signal.

This also clarifies the earlier topology-versus-feature-divergence statement. Among
same-pipeline Twitter graphs, graph-level topology and feature-cloud distances are
weakly related; the stronger all-graph association is driven by the political
graphs' joint topology and embedding-pipeline shift. That is a separate scale from
the node-pair result here, but both argue against treating topology and feature
geometry as one scalar axis.

## Caveats and next test

- Short-distance pairs are exact but random-walk sampled. Matching distances to the
  same anchor controls node marginals substantially, yet does not make pairs uniform
  over all pairs at a given distance.
- The far endpoint is `>3_or_disconnected`, not a known numeric path length; only
  the 1–3 correlation uses exact lengths.
- The uniform-pair extension computes exact lengths for connected pairs and treats
  disconnected pairs as a separate outcome; it does not encode disconnection as an
  invented distance such as 1,000.
- The primary analysis conditions on all five nodes having nonzero text features,
  isolating embedding geometry. Missing-text patterns may provide additional signal.
- The adjacent-versus-far probe can exploit absolute feature regions associated
  with neighbor-sampled/high-degree nodes, not only feature similarity. That is
  relevant to what aggregation can access, but should not be called homophily.
- Embedding coordinates are arbitrary. Recurrent dimensions 4 and 128 across several
  organic graphs are reproducible indices, not intrinsically interpretable concepts.
- One sampling seed was run. Cross-graph consistency is encouraging, but confidence
  intervals or a seed sweep would be required for a paper-level uncertainty claim.

The decisive mechanism test is small: for each graph, permute the selected 1/4/16
dimensions across nodes (preserving every marginal but breaking structure coupling),
then rerun NM inference or the feature-only probe. A selective performance drop would
show that the low-dimensional signal is not merely detectable but actually used.
