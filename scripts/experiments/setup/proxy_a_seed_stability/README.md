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

## 40,000-node repeat (2026-09-09)

Use `--samples 40000` with the same runner revision `020c7f39`, seeds, classifiers,
and policies. Successful-launch output directory is
`/dataMeR1/phil/gfm/prodigy-proxy-a-seeds/log/proxy_a_seed_stability_40k_retry01`.
The first attempt (`..._40k`) failed before sampling because tmux inherited an
activated environment with base Python ahead on PATH. Use the explicit
`/home/mhchu/miniconda3/envs/prodigy/bin/python` executable after activation.
Do not rerun into either existing directory.

Download completed small outputs into the analysis leaf's
`sample_40000/data/`. Run `analyze.py --run-dir <analysis-leaf>/sample_40000`, then
`compare_sizes.py`. The original 4k outputs stay intact. At 40k, the historical
sorted sampler exhausts candidate populations for some smaller graphs, so its
sampled IDs may be identical across seeds; its remaining variability is then
classifier splitting. Uniform sampling remains the primary estimator.
