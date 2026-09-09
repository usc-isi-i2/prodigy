# Does proxy-A explain compatibility beyond donor strength?

Using 40k uniform proxy-A averaged over three estimator seeds and fixed three-training-seed NM AUC, exclude all diagonal cells. Regress both quantities on source and target fixed effects with least squares on the 72 observed foreign cells (not naive double-centering of a matrix with missing diagonal).

- Residual Pearson r: **−0.576**; residual Spearman: **−0.320**.
- Proxy-A accounts for 33.2% of the remaining in-sample squared variation after the additive baseline. This is not held-out explained variance or a causal fraction.
- Estimator-seed partial Pearson r: −0.572, −0.579, −0.576.
- Leaving each graph out in both roles and refitting gives partial r from −0.703 to −0.156. Removing Election weakens it most.
- Exploratory removal of just the two Election↔COVID Political cells, refitting all effects, reduces partial r to **−0.196**. Those two cells carry substantial leverage; they are not discarded from headline analyses.

## Held-out graph prediction

For each target, omit every cell involving that graph in either role. Fit source+target effects, with and without one linear proxy-A coefficient, on the remaining eight graphs. Predict relative donor performance on the held-out target. Its intercept is unknown; center predictions and observed outcomes only to score relative errors. Ranking and regret require no centering. This uses observed performance of known donors on other graphs, not only raw donor descriptors; it does not claim absolute AUC prediction on an unseen target.

| Mean over nine held-out targets | Source effects | Source effects + proxy-A |
|---|---:|---:|
| Donor-rank Spearman | 0.823 | 0.876 |
| Target-centered RMSE | 0.0294 | 0.0257 |
| Best-donor selection regret | 0.00573 | 0.00881 |

RMSE improves on 5/9 targets. Selection changes only on Facebook, where it worsens. No cell-IID significance tests or causal inference: only nine graphs, shared episodes, and correlated source properties. The pair-removal check is an exploratory influence diagnostic.

**Conclusion:** proxy-A carries incremental compatibility information, but it is uneven and strongly influenced by the Election/Political relationship. It is not established as a general mechanism or universally better donor selector.

Reproduce with `check_incremental.py` in the parent folder. Data tables preserve every cell and fold. Worktree `/Users/philipp/projects/gfm/prodigy-proxy-a-seeds`; branch `codex/proxy-a-seeds-clean-20260909`. No new Tucker jobs.
