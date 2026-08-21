# GraphSAGE mixture-scaling results

## Scope and protocol

These tables report plain one-layer, 256-dimensional GraphSAGE trained with native
random-walk/edge negative-sampling SSL. Every downstream result uses the identical
fixed split of ten labeled nodes per class and reports linear-probe macro one-vs-rest
AUC, macro F1, and accuracy. SSL checkpoints are never selected using downstream
performance.

The held-out ladder uses one declared prefix order. For each target, that target is
removed before constructing the source prefix. Its six rungs are assembled as follows:

- `k=1`: the first non-target specialist, evaluated through the full transfer matrix;
- `k=2..5`: dedicated intermediate-mixture runs;
- `k=6`: the leave-one-target-out mixture.

Thus, every ladder score is genuinely held out from its target. Adaptation efficiency
is the normalized trapezoidal area under downstream score versus
`log10(checkpoint_step)` at steps 100, 300, 900, and 2,500. Higher is better.

## RQ1: fixed-order held-out mixture ladder (seed 0)

| Sources | AUC efficiency | F1 efficiency | Accuracy efficiency | AUC @ 2.5k | F1 @ 2.5k | Accuracy @ 2.5k |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.7576 | 0.5833 | 0.6006 | 0.7740 | 0.6114 | 0.6254 |
| 2 | 0.7570 | 0.5789 | 0.5960 | **0.7778** | 0.6088 | 0.6207 |
| 3 | 0.7438 | 0.5606 | 0.5792 | 0.7566 | 0.5798 | 0.5954 |
| 4 | 0.7524 | 0.5701 | 0.5904 | 0.7599 | 0.5827 | 0.5982 |
| 5 | **0.7645** | **0.5852** | 0.6051 | 0.7702 | 0.5957 | 0.6106 |
| 6 | 0.7640 | **0.5852** | **0.6052** | 0.7758 | 0.6025 | 0.6194 |

The relationship is not monotonic. Averaged over seven targets, increasing from one
to six sources changes adaptation efficiency by only +0.0064 AUC, +0.0019 macro F1,
and +0.0046 accuracy. Six sources improve over one on 5/7 targets for AUC and 6/7 for
F1 and accuracy, but the best mixture size depends strongly on the target. The most
consistent pattern is a dip at three sources followed by recovery at five to six;
five sources slightly leads six on mean AUC efficiency, while they tie on F1.

This is evidence against a simple “more sources is always better” law. Because the
ladder adds sources in one fixed order, mixture size and source identity are still
confounded. It answers breadth along this declared trajectory, not the causal effect
of semantic diversity at fixed mixture size. Multiple orders or matched-size mixture
subsets are required for that stronger claim.

## Three-seed endpoint comparison at step 2,500

The table below is the mean paired difference `leave-one-out minus target specialist`.
The target specialist was pretrained on the evaluation graph; the leave-one-out model
was not, so this is an intentionally demanding transfer comparison rather than a pure
mixture-size contrast.

| Target | AUC delta | F1 delta | Accuracy delta |
|---|---:|---:|---:|
| Cora | -0.0131 | -0.0468 | -0.0196 |
| COVID political | -0.0598 | -0.0625 | -0.0572 |
| Election 2020 | +0.0119 | +0.0385 | +0.0378 |
| Facebook page reference | +0.0231 | -0.0170 | +0.0047 |
| PubMed | -0.0009 | +0.0052 | -0.0148 |
| TwiBot-20 | +0.0161 | -0.0066 | -0.0203 |
| Ukraine/Russia suspended | -0.0161 | -0.0102 | -0.0196 |
| **Mean** | **-0.0056** | **-0.0142** | **-0.0127** |

Across three seeds, the direction is stable for most target/metric pairs, but it is
heterogeneous across targets. Leave-one-out wins all three seeds for Election 2020 on
all metrics, and for Facebook AUC/accuracy and TwiBot-20 AUC. The target specialist
wins all three seeds for Cora, COVID political, and Ukraine/Russia on all metrics.

## Evidence inventory

- `primary_s0/primary_results.csv`: 56 rows (14 models × one target × four checkpoints).
- `primary_s1_s2/primary_results.csv`: 112 rows (28 models × one target × four checkpoints).
- `matrix_s0/matrix_results.csv`: 392 rows (14 models × seven targets × four checkpoints).
- `ladder_s0/ladder_results.csv`: 112 rows (28 target/mixture pairs × four checkpoints).
- `analysis/heldout_ladder.csv`: the reconstructed leakage-free 1–6-source ladder.
- `analysis/adaptation_by_target.csv` and `analysis/adaptation_by_size.csv`: adaptation-efficiency tables.
- `analysis/primary_summary.csv` and `analysis/primary_contrasts.csv`: three-seed endpoint summaries.
- `analysis/matrix_step2500.csv`: the complete 14 × 7 transfer matrix at step 2,500.
- `analysis/audit.json`: input hashes, row counts, and analysis invariants.

Raw evaluation shards, logs, and checkpoints remain preserved on Tucker under
`/dataMeR1/phil/gfm/mixture-scaling`; interrupted runs were retained separately and
were not deleted.
