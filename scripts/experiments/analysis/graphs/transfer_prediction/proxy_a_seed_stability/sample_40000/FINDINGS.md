# Proxy-A: three-seed stability

Raw features, 40,000 nonzero nodes per graph. Joint node-sampling and classifier-split variation. Fixed NM transfer outcomes.

## legacy_sorted
- Within-target mean Spearman for seeds 0/1/2: -0.7011, -0.6746, -0.7275
- Mean / maximum pairwise accuracy SD: 0.186 / 0.440 percentage points.

## uniform
- Within-target mean Spearman for seeds 0/1/2: -0.7249, -0.7063, -0.7222
- Mean / maximum pairwise accuracy SD: 0.207 / 0.529 percentage points.

Mean absolute sampling-policy shift in domain accuracy: 1.803 percentage points.

Historical sorted-candidate sampling favors low node IDs. Both policies use new random streams; legacy seed 0 is not an exact historical replay. Three estimator seeds are not training replications or a confidence interval. No inference of causal transfer effects.
