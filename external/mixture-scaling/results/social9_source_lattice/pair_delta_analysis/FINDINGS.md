# Vanilla GraphSAGE Social-9 source-composition findings

Last updated: 2026-09-14.

## Scope

This analysis consolidates the completed seed-0 Social-9 native link-prediction
GraphSAGE lattice. The source lattice contains 54 models: nine single-graph
specialists, all 36 unordered source pairs, and nine leave-one-out mixtures.
The encoder is the same one-layer 768-to-256 GraphSAGE throughout. Checkpoints
are selected using source-side SSL validation, never downstream labels.

The frozen encoders have 270 node-classification evaluations (54 models by five
targets) and 324 repaired static-link evaluations (54 models by six targets).
The experiment completed on 2026-09-06; the preserved aggregate tables were
refreshed on 2026-09-11.

`social9_graphsage_matrix_pairs_loo.tsv` joins both downstream tasks into 594
rows and adds composition labels, the excluded graph for leave-one-out models,
training metadata, and matched singleton comparisons.

## Definitions

Two mixture contrasts answer different questions and must remain separate.

1. **Versus mean constituent** asks whether mixing is better than choosing a
   constituent at random. This measures diversification or robustness.
2. **Versus better constituent** asks whether mixing beats an oracle choice of
   the stronger constituent for the same target. This measures selection value.

The directional pair matrices use a third, asymmetric contrast:

`delta(base, added, target) = AUC(pair, target) - AUC(base, target)`.

Each displayed cell averages this delta over held-out classification targets,
excluding targets equal to either pair source. Unless a figure explicitly says
otherwise, the five-target evaluation panel remains fixed when rows and columns
are pruned. Keeping the target panel fixed is necessary for comparable marginal
means.

## Main results

### Mixtures hedge source choice but rarely beat the oracle singleton

| Evaluation | Pair vs mean constituent | Pair win rate | Pair vs better constituent | Pair win rate |
|---|---:|---:|---:|---:|
| Classification | +0.00254 AUC | 67.8% | -0.01120 AUC | 31.1% |
| Static link prediction | +0.02436 AUC | 86.1% | -0.00632 AUC | 30.1% |

Pairs usually improve on the average constituent, particularly for native link
prediction, but usually lose to the target-specific better constituent.

Leave-one-out mixtures show the same distinction more strongly:

| Evaluation | LOO vs mean included singleton | LOO win rate | LOO vs best included singleton | LOO win rate |
|---|---:|---:|---:|---:|
| Classification | +0.00093 AUC | 53.3% | -0.02787 AUC | 2.2% |
| Static link prediction | +0.04456 AUC | 100.0% | -0.02054 AUC | 0.0% |

### Breadth improves native LP much more than classification

| Composition | Mean classification AUC | Mean static-LP AUC |
|---|---:|---:|
| Singleton | 0.75693 | 0.70379 |
| Pair | 0.75947 | 0.72815 |
| Leave one out | 0.75785 | 0.74835 |

The strong static-LP rise does not carry through to average classification. On
the two targets shared by both evaluation panels, source-composition rankings
are only weakly aligned: Pearson correlation is 0.143 for Facebook Pages and
-0.099 for TwiBot-20. Native LP quality alone is therefore not a reliable model
selection rule for classification in this evidence.

### Pair effects depend strongly on the target

Mean pair-minus-better-singleton classification effects are -0.01796 AUC for
COVID Political, -0.00235 for Election 2020, -0.00832 for Facebook Pages,
-0.02342 for TwiBot-20, and -0.00396 for Ukraine/Russia Suspended. Election is
nearly neutral and pairs win 52.8% of its cells; TwiBot has the largest average
penalty.

### Directional addition matrix

Across all 72 off-diagonal directions in the fixed-panel matrix, adding a graph
to the row singleton changes held-out classification AUC by +0.44 percentage
points on average. Column means identify additions that generally strengthen a
weaker base: TwiBot-20 +1.72 points, COVID Political +1.49, and Midterm +1.28.
Election 2020 is the strongest negative addition at -1.87 points. This positive
directional mean is compatible with the negative better-constituent contrast:
the pair can improve the weaker base without surpassing its stronger member.

## Subset searches

All subset searches enumerate every possible retained source set and use strict
inequalities on the off-diagonal directional cell means.

### Unique largest all-positive subset

The unique largest source subset with every directional cell greater than zero
has five graphs:

- COVID-19 Twitter
- Midterm
- TwiBot-20
- Ukraine/Russia Suspended
- Ukraine/Russia Twitter

Its 20 directional cells range from +0.048 to +2.28 percentage points, with a
mean of +0.896 points. The weakest cell rounds to +0.05, so the strict result has
little margin and should not be interpreted as statistically established under
one training seed.

### Largest all-negative subsets

No three-graph source subset is entirely negative. Two two-graph subsets have
both directions below zero:

| Pair | Direction 1 | Direction 2 | Directional mean |
|---|---:|---:|---:|
| Election 2020 and UKR/RUS Twitter | -0.76 pp | -2.70 pp | -1.73 pp |
| CP/HK Twitter and UKR/RUS Twitter | -0.38 pp | -0.59 pp | -0.48 pp |

The figure uses the stronger negative pair, Election 2020 and UKR/RUS Twitter.

## Relation to the older held-out ladder

Ten exact source-set/target matches exist between the older six-rung held-out
ladder and the Social-9 lattice. Their AUC correlation is 0.969 and mean absolute
difference is 0.0359. For the exact singleton-to-pair contrast, four of five
targets agree in direction. TwiBot reverses sign. The two studies both find a
slightly negative average pair effect, but the ladder's rungs three through six
have no exact Social-9 counterparts. The ladder is supporting evidence, not a
protocol-identical extension of this lattice.

## Interpretation and limits

The supported conclusion is that broader vanilla GraphSAGE pretraining improves
the native LP objective and reduces the risk of choosing a poor source, but does
not reliably beat the best target-specific specialist or improve downstream
classification on average.

This is a single-training-seed descriptive lattice. Cells share models, targets,
and evaluation episodes, so they are not independent replicates. Small positive
cells, column rankings, and strict subset membership require seed replication
before inferential claims. The all-positive and all-negative subsets are
post-hoc searches over the observed matrix and should be labeled exploratory.

## Recommended next analysis

1. Replicate the nine specialists, 36 pairs, and selected LOO models over at
   least three seeds.
2. Make held-out-target contrasts the primary analysis and report both mean-
   constituent and better-constituent baselines.
3. Estimate target-by-source inclusion effects rather than one global source
   ranking.
4. Test whether source-only validation or performance on other targets can
   select the best singleton or pair without target-label access.
5. Fill a small, pre-registered set of three-to-seven-source mixtures chosen
   from positive and negative pair interactions to connect pairs with LOO.
6. Treat native LP and downstream classification as separate outcomes unless a
   broader aligned-target analysis establishes predictive correspondence.

## Files

- `social9_graphsage_matrix_pairs_loo.tsv`: joined 594-row analysis table.
- `social9_graphsage_pair_delta_heatmap.png`: full directional 9-by-9 matrix.
- `social8_graphsage_pair_delta_heatmap_no_election.png`: exploratory matrix
  excluding Election as both source and target.
- `social7_graphsage_pair_delta_heatmap_no_election_no_covid_political.png`:
  exploratory matrix excluding Election and COVID Political as sources and
  targets.
- `graphsage_all_positive_pair_delta_heatmap.png`: unique largest all-positive
  source subset with the fixed five-target panel.
- `graphsage_all_negative_pair_delta_heatmap.png`: stronger of the two largest
  all-negative source subsets with the fixed five-target panel.
