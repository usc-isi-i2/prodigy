# Strong raw-feature MLP baseline on TwiBot-20

## Evidence contract

This supervised baseline uses raw features only and a deterministic 60/20/20
TwiBot-20 split. Configuration selection used validation data on seed 0, after
which the configuration was locked for five seeds and five label budgets. The
producing Tucker worktree was `prodigy-strongmlp` at `bec6dcaf66`. The committed
protocol, selection record, 25/25 result cells, and per-cell curves are under
`data/`.

## Results

| labels per class | mean test ROC-AUC | sample SD |
|---:|---:|---:|
| 10 | 0.5511 | 0.0431 |
| 100 | 0.6379 | 0.0184 |
| 500 | 0.6915 | 0.00536 |
| 1,000 | 0.7163 | 0.00398 |
| full training split | 0.7469 | 0.00401 |

Mean macro-F1 over the same budgets is 0.5289, 0.5995, 0.6321, 0.6556, and
0.6822. Performance improves monotonically with label budget, reaching test
ROC-AUC 0.7469 +/- 0.0040 on the full training split.

This is a strong TwiBot-20 raw-feature baseline only. It supports neither a
cross-target claim nor a topology claim. The five seeds vary label downsampling
and training; they do not resample evaluation episodes.
