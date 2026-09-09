# Proxy-A: three-seed stability

Raw features, 4,000 nonzero nodes per graph. Joint node-sampling and classifier-split variation. Fixed NM transfer outcomes.

## legacy_sorted
- Within-target mean Spearman for seeds 0/1/2: -0.6852, -0.6772, -0.7083
- Mean / maximum pairwise accuracy SD: 0.705 / 1.655 percentage points.

## uniform
- Within-target mean Spearman for seeds 0/1/2: -0.7169, -0.6818, -0.7003
- Mean / maximum pairwise accuracy SD: 0.530 / 1.381 percentage points.

Mean absolute sampling-policy shift in domain accuracy: 2.885 percentage points.

Historical sorted-candidate sampling favors low node IDs. Both policies use new random streams; legacy seed 0 is not an exact historical replay. Three estimator seeds are not training replications or a confidence interval. No inference of causal transfer effects.
