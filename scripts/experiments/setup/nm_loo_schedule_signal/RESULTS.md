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
