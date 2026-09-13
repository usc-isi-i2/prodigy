# Paper mechanism sweeps

This completed analysis tests graph-mixing ratio, fixed-composition pretraining
scale, and encoder capacity. It contains 810 NM cells, 375 classification cells,
90 physical checkpoints, and 75 models compared across tasks. The training seed
is the replication unit; the observed three-seed range is not a confidence interval.

The decision rules require a 0.001 ROC-AUC practical margin and no seed-mean
target regression worse than -0.001. See [FINDINGS.md](FINDINGS.md) for the result.
