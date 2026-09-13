# Raw-feature proxy-A seed stability

Three independent estimator seeds (0/1/2), 4,000 nonzero-feature nodes per graph,
all nine graphs, 36 unordered pairs per seed. Fit the historical logistic domain
classifier with unchanged hyperparameters. Repeat both historical sorted-candidate
sampling and uniform nonzero-node sampling: the historical sampler truncates
sorted candidates and therefore favors low node IDs. This is a sampling-policy
sensitivity control, not a change to historical results.

Seeds vary feature samples and held-out classifier splits together; this does
not separate the two variance components. Use identical pair split seeds across
policies. Matrices are symmetric by construction; no independent reverse fits.
These are new estimates, not bit-exact historical seed-zero reproductions.
Keep source/target NM outcomes fixed at their existing three-training-seed means.
Report sample SD across estimator seeds, not a confidence interval. Three seeds
do not establish convergence with respect to sample size.

CPU-only Tucker run in an isolated worktree and tmux. Activate `prodigy`, with
conda bin on PATH and its lib in LD_LIBRARY_PATH. Limit BLAS/OpenMP to four threads.

```
python scripts/experiments/setup/proxy_a_seed_stability/run.py --out /dataMeR1/phil/gfm/prodigy-proxy-a-seeds/log/proxy_a_seed_stability --dry-run
python scripts/experiments/setup/proxy_a_seed_stability/run.py --out /dataMeR1/phil/gfm/prodigy-proxy-a-seeds/log/proxy_a_seed_stability
```

The output retains sampled IDs/features, input metadata/hashes, protocol, six
matrices and a DONE marker. Convergence warnings are fatal. Analysis belongs in
`analysis/graphs/transfer_prediction/proxy_a_seed_stability/`.
