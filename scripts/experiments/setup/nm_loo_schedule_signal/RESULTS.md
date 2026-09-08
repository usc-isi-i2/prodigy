# Results: LOO exposure x source-block signal experiment

All 18 training runs and all 18 fixed-stream evaluations completed successfully on
2026-09-07. Evaluation used the archived final-core Ukraine stream with episode-plan
fingerprint `c7161b74e2e7b97ecb2e65e6cea6bffb594aee907cac538849fdbcb8a8d7f49c`.
The evaluated checkpoint is step 2,500 for every run.

| Exposure | Block size | Accuracy mean +/- SD | Macro-F1 mean +/- SD | ROC-AUC mean +/- SD |
|---|---:|---:|---:|---:|
| node-proportional | 1 | **0.36412 +/- 0.00495** | **0.36412 +/- 0.00490** | **0.88554 +/- 0.00204** |
| node-proportional | 16 | 0.32815 +/- 0.05390 | 0.32813 +/- 0.05384 | 0.86596 +/- 0.03000 |
| node-proportional | blocked | 0.34805 +/- 0.00507 | 0.34804 +/- 0.00507 | 0.87161 +/- 0.00657 |
| uniform-over-graphs | 1 | 0.33280 +/- 0.00489 | 0.33280 +/- 0.00491 | 0.87529 +/- 0.00207 |
| uniform-over-graphs | 16 | 0.23865 +/- 0.01588 | 0.23857 +/- 0.01585 | 0.82355 +/- 0.01075 |
| uniform-over-graphs | blocked | 0.29765 +/- 0.02321 | 0.29762 +/- 0.02325 | 0.85181 +/- 0.01181 |

Node-proportional exposure beats uniform exposure at every tested block size, and
`k=1` is best under both exposure rules. The predicted uniform-plus-`k=1` optimum is
therefore rejected for this mixture. The schedule response is not monotone: `k=16`
is worse than fully blocked under both exposure rules. Node-proportional `k=16` is
also unstable because seed 1 falls to 0.26598 accuracy while seeds 0 and 2 reach
0.35661 and 0.36185. These results provide the requested signal; do not infer that a
larger block-size sweep is warranted without first explaining or replicating the
medium-block failure.

Training receipts and checkpoints live on Tucker under
`/dataMeR1/phil/gfm/prodigy-loosignal/log/nm_loo_schedule_signal/shared_20260908_signal2`.
Evaluation JSONs live under
`/dataMeR1/phil/gfm/prodigy-loosignal/log/nm_loo_schedule_signal_eval/results`.
The exact per-run values are recorded in `data/results.tsv`.

## All-target `k=1` cross-evaluation

The three uniform and three node-proportional `k=1` checkpoints were subsequently
evaluated on all nine final-core targets with the same frozen 512-episode streams.
All 54 cells completed with exit code 0 at evaluation revision `8c78dc37`. Exact
per-seed accuracy, macro-F1, ROC-AUC, and both episode fingerprints are committed in
`data/all_targets_k1_results.tsv`.

The table reports paired node-proportional minus uniform ROC-AUC differences as the
mean +/- sample SD over the three matched training seeds.

| Target | Uniform AUC | Node-proportional AUC | Paired AUC difference +/- SD |
|---|---:|---:|---:|
| covid | 0.94187 | 0.96079 | +0.01893 +/- 0.00283 |
| midterm | 0.85330 | 0.85645 | +0.00315 +/- 0.01123 |
| covid_political | 0.84760 | 0.83835 | -0.00925 +/- 0.01507 |
| election2020 | 0.84200 | 0.83610 | -0.00591 +/- 0.00947 |
| ukr_rus_suspended | 0.86937 | 0.82084 | -0.04854 +/- 0.00694 |
| twibot20 | 0.89634 | 0.90225 | +0.00591 +/- 0.00490 |
| cp_hk | 0.71037 | 0.71757 | +0.00721 +/- 0.00901 |
| facebook_page_reference | 0.84796 | 0.83887 | -0.00909 +/- 0.00342 |
| held-out ukr_rus | 0.87529 | 0.88554 | +0.01026 +/- 0.00284 |
| Macro average | 0.85379 | 0.85075 | -0.00304 +/- 0.00401 |

The aggregate exposure effect is small and inconclusive at three seeds: macro AUC
changes by only `-0.00304 +/- 0.00401`, while macro accuracy changes from 0.25294 to
0.25482. Exposure nevertheless redistributes performance across individual graphs.
Node-proportional sampling assigns 95.09% expected exposure to `covid` rather than
12.5% under uniform sampling; COVID gains 0.01893 AUC, while the most starved graph,
`ukr_rus_suspended` (0.30% rather than 12.5%), loses 0.04854 AUC. Across the eight
training sources, expected exposure change correlates with accuracy change at
Pearson `r=0.843` and with AUC change at `r=0.481`. These are descriptive correlations
over only eight graphs and are dominated by the extreme COVID imbalance.

The held-out `ukr_rus` graph has zero direct exposure under both rules but gains
0.01026 AUC under node-proportional training, consistent with transferred benefit
from the heavily weighted source mixture rather than direct target exposure. The
supported conclusion is therefore not that either exposure rule universally wins:
uniform exposure slightly protects aggregate cross-graph AUC, while proportional
exposure shifts capacity toward COVID and improves held-out Ukraine in this mixture.
