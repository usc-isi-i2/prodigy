# Node-only MLP versus neighborhood-aware encoders

## Result

Fixed neighborhood context usually improves cross-graph link prediction, but the evidence does not establish a large general advantage from learned message passing itself.

The matched Social-9 input-view study compares three independently trained MLP encoders: node features only, incoming-neighbor mean only, and node features concatenated with the incoming-neighbor mean. The LP comparison covers nine training sources and six targets at seed 0, for 54 matched source-target cells.

| Input view | Mean ROC AUC | Difference from node only | Wins over node only |
|---|---:|---:|---:|
| Node only | 0.6101 | — | — |
| Neighborhood only | 0.6393 | +2.92 pp | 37/54 |
| Node + neighborhood | 0.6619 | +5.18 pp | 43/54 |

The benefit is stronger off the diagonal. Neighborhood-only improves held-out cells by 3.59 points on average and wins 34/48. Node + neighborhood improves held-out cells by 5.71 points and wins 39/48.

The gains remain target-specific. For node + neighborhood, the mean source-averaged changes are +10.41 points on TwiBot, +8.25 on COVID-19, +7.55 on UKR/RUS, +3.65 on Midterm, +1.92 on CP/HK, and -0.72 on Facebook. Neighborhood-only is more uneven: Facebook falls by 7.59 points and CP/HK by 0.81, while TwiBot gains 10.67.

## What “neighborhood” means here

This comparison does not use learned message passing. For each sampled root, it computes the mean feature vector of incoming sampled neighbors, excluding explicit self-loops. The neighborhood-only MLP receives that fixed mean. The node + neighborhood MLP receives the concatenation of the root feature and the fixed mean. The configured fanout is 15.

The result therefore shows that local graph-selected context carries useful transfer signal. It does not by itself show that GraphSAGE-style learned aggregation is necessary.

## Direct MLP versus GraphSAGE pilot

A separate HK link-prediction pilot matched the MLP and one-hop GraphSAGE on the same 100,000 training edges and nonedges, 2,500 updates, saved validation/test pairs, cosine scoring, and seed 0. Test ROC AUC was 0.6762 for the MLP and 0.6849 for GraphSAGE, a 0.87-point GraphSAGE advantage.

That advantage is concentrated in pairs with at least one zero-feature endpoint: GraphSAGE scores 0.6689 versus 0.6063 for the MLP. When both endpoints have features, the MLP is stronger, 0.7024 versus 0.6908. The overall difference is a one-graph, one-seed pilot and does not establish a robust architectural winner.

## Interpretation

Neighborhood information helps primarily as a coverage mechanism for missing or weak endpoint features and as a transferable summary of local context. The three-view study shows a substantial average gain from exposing that information to an MLP. The HK pilot shows only a small overall increment from learned one-hop GraphSAGE over a trained node-only MLP, with the sign depending on feature availability.

The combined evidence supports three narrower claims:

1. Node-only representations leave useful local-context signal unused.
2. A fixed neighbor summary recovers much of that signal without message passing inside the encoder.
3. Learned message passing is especially useful when endpoint features are absent, but is not uniformly better when endpoint features are present.

## FP is target-dominated; LP has source-target structure

A balanced two-way decomposition sharpens the contrast between the two objectives. For each input view, predict every matrix cell using only its target mean, only its source mean, or the additive source and target means. Report the fraction of total cell variance explained without fitting interaction terms.

| Input view | Objective | Target-only R² | Source-only R² | Additive source + target R² |
|---|---|---:|---:|---:|
| Neighborhood only | FP | 0.9935 | 0.0019 | 0.9953 |
| Node + neighborhood | FP | 0.9965 | 0.0019 | 0.9984 |
| Neighborhood only | LP | 0.6034 | 0.2253 | 0.8287 |
| Node + neighborhood | LP | 0.3908 | 0.4189 | 0.8097 |

FP is almost entirely target-dependent in these matrices. Across training sources, the source-mean range is only 0.0110 for neighborhood-only FP and 0.0138 for node + neighborhood FP, compared with target-mean ranges of 0.2433 and 0.2374. The evaluation graph determines nearly all observed variation in reconstruction error.

LP is not target-independent. Rather, it contains substantial variation along both axes. Target identity alone explains 39–60% of LP variance, while source identity alone explains 23–42%. For node + neighborhood LP, the source main effect is slightly larger than the target main effect. This matches the donor, pair, and ladder results: the relation between node features and connectivity changes across graphs, so the training source and its compatibility with the target matter.

The concise conclusion is: **FP transfer is dominated by target difficulty; LP transfer contains strong source-target transfer structure.** The R² values are descriptive decompositions of these balanced seed-0 matrices, not out-of-sample predictive estimates or causal variance components.

## Limits

- The broad matrix comparison is seed 0 and changes input dimensionality across views.
- The variance decomposition uses matrix cell variation and does not quantify uncertainty across seeds.
- Checkpoint selection is view-specific, so it is a system comparison rather than an isolated information intervention.
- The LP matrix covers six targets, although all nine graphs appear as training sources.
- These historical runs predate the repaired Suspended artifact audit; results involving that source require the existing provenance caveat.
- The HK GraphSAGE comparison is a single graph, seed, and sampled pair set.

## Provenance

The three-view tables are in `results/context_mlp_transfer/aggregated/`; the overview figure is preserved beside this note. Node-only LP values come from `results/mlp_transfer_explains_ladder/data/single_source.tsv`. The matched HK pilot is documented in the Prodigy analysis tree at `scripts/experiments/analysis/evaluation/lp_baselines/hk_lp_baselines/FINDINGS.md`.
