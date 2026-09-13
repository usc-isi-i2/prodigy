# Joint feature MMD: seven graphs, September 11, 2026

Results: `data/feature_joint_mmd_seven.json`; plot: `figures/feature_joint_mmd_seven.png` (also PDF). Render with `plot_feature_joint_mmd.py`.

Election and COVID Political are excluded. We use the original KS cached samples for the other graphs, replacing Suspended with the verified repaired v2 artifact (SHA256 18baf3fa4e49b767eb1a0f7b670a8aa39ac16e2de0a991ca75a90685496e1cd4). These are raw 768-dimensional bio features, not trained MLP outputs. Exact all-zero rows are excluded; no feature normalization or projection is applied.

Compute exact unbiased MMD squared on 2,000 vectors per graph, averaging RBF kernels with bandwidths 0.5, 1 and 2 times a shared median Euclidean distance estimated from 300 vectors per graph. Global bandwidths permit pair comparisons. Repeat three times using subsets of the fixed 20,000-row caches. Within a repeat each graph has a disjoint second 2,000-row sample for a same-graph baseline. Diagonals display these baselines rather than self-distances. Unbiased estimates can be slightly negative; raw values are preserved. The color scale starts at zero. Repeat standard deviations quantify this subsampling variability, not a population confidence interval; no formal hypothesis test was performed. Shared users between datasets may contribute to similarity.

The RBF kernel compares whole feature vectors, including dependence across coordinates; this is a sampled joint-distribution comparison, not an exhaustive comparison of every node. MMD scale depends on bandwidth and is not comparable numerically to KS.

UKR/RUS–COVID is closest: 0.000634, repeat SD 0.000195. Facebook pairs range from 0.019286 (UKR/RUS) to 0.029092 (Midterm), larger than every non-Facebook pair (maximum 0.011355). All off-diagonal mean distances exceed the observed same-graph baseline magnitudes: the maximum absolute baseline over all repeats is 0.0000499. The mean diagonal ranges from -0.0000284 to 0.00000886.

Compute script: mixture-scaling branch `codex/feature-dimension-ks`, commit `19d6312`, `scripts/feature_joint_mmd.py`. Local worktree `/tmp/feature-dimension-ks`; Tucker worktree `/dataMeR1/phil/gfm/mixture-scaling-feature-dimension-ks`. Output `/dataMeR1/phil/gfm/mixture-scaling/results/feature_joint_mmd_seven_s2026/results.json`. CPU-only runtime 110.84 seconds. Figure and summary prepared in the local PRODIGY worktree `/Users/philipp/projects/gfm/prodigy`.
