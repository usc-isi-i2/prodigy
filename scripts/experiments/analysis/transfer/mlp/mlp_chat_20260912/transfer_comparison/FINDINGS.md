# Does single-source transfer explain the MLP ladder?

The historical cosine-AUC ladder resembles the best available single-source specialist more than the average specialist. This is descriptive evidence for source coverage and retention, not causal identification or proof of complementary representations.

## Method

Compare the original node-only LP specialists (9 sources × 6 targets) with the convergence ladder on exactly those six targets. Both tables report validation-oriented cosine ROC-AUC. Do not substitute the separately computed raw-dot AUC/BCE. Each ladder rung is a separate model, not sequential fine-tuning.

For each target, anchor predictions at its observed rung-1 AUC. Add either the change in mean specialist AUC or the change in maximum specialist AUC over the cumulative source set. No coefficients are fitted. A flat rung-1 prediction is the reference. Evaluate rungs 2–9 (48 cells), and separately rungs 2–5 (24 cells) before corrupted Suspended enters training. Maximum specialist selection is a retrospective target-performance oracle, not an independently validated deployable selection rule. The cells are dependent; correlations are descriptive, with no IID significance claim.

| Prediction | Full historical RMSE (AUC points) | Rungs 2–5 RMSE |
|---|---:|---:|
| Flat at rung 1 | 2.98 | 2.20 |
| Mean specialist gain | 6.88 | 4.59 |
| Best specialist gain | 2.08 | 1.31 |

Best-specialist gain reduces squared error versus flat by 51.6% overall and 64.6% before Suspended. Its correlation with cumulative observed gains is 0.883 / 0.855, and with adjacent-rung gains 0.932 / 0.847. Mean-specialist predictions substantially overstate degradation: mixture performance does not behave like averaging specialist AUCs (nor is that average an ensemble AUC).

## What this explains and misses

Large jumps occur when strong specialists become available, especially when the target itself enters training (COVID rung 2, Midterm rung 3, TwiBot rung 7, Facebook rung 9). Later declines on UKR/COVID/Midterm are not captured by a monotone best-specialist curve. HK's small increase after COVID is compatible with transfer, but its fluctuations are not fully explained.

Restricting to still-unseen targets weakens the result: best-versus-flat RMSE is 1.67 versus 1.80 points across 19 historical cells, and 1.12 versus 1.39 across 13 pre-Suspended cells. Thus target inclusion drives much of the overall result. Every rung-9 target is 1.42–3.16 AUC points below its best single-source specialist in absolute performance. This does not demonstrate mixture synergy beyond the retrospective specialist oracle.

## Limits and next experiment

Specialist selected checkpoints are 7k–10k steps; ladder selected checkpoints are much later. Seed/order are singular. Rungs 6–9 trained on known corrupted Suspended features; all historical downstream claims involving those models remain provisional. The rungs 1–5 restriction avoids that particular training corruption but not the budget confound. The other three ladder targets lack matching specialist evaluations in the available table. BCE cannot be analyzed from this AUC-only specialist table.

A stronger test is to train specialists to the same operational convergence standard on repaired data, evaluate all nine targets with matching raw-dot AUC/BCE, pre-register the mean/best/coverage predictors, and test them on additional source orders. Separate target-held-out gains from gains upon adding the target itself. No new training was launched for this analysis.

## Reproduce and provenance

Run `OPENBLAS_NUM_THREADS=1 MPLBACKEND=Agg MPLCONFIGDIR=/tmp/mlp-mpl /opt/homebrew/bin/python3.11 analyze.py` in this directory. Data were read on 2026-09-11 from:

- Tucker `/dataMeR1/phil/gfm/mixture-scaling-node-only/results/node_only_transfer/aggregated/node_mlp_lp_transfer.tsv`
- Tucker `/dataMeR1/phil/gfm/mixture-scaling/results/node_mlp_ladder_s0_convergence_recovery/raw/ladder_results.csv`

`data/` contains exact input tables plus cell-wise comparisons; `summary.json` retains metrics. Shaded plot regions mark affected rungs 6–9. Existing source results were not changed.
