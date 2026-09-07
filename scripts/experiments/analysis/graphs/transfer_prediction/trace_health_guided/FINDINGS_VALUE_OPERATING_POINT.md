# Value replacement improves ranking but collapses classification

Read-only audit of the completed Hong Kong-to-covid_political 50k K/V run,
2026-09-07. This file concerns the exact `classrefkv_long_20260907` exports, not
an average across earlier context-removal experiments. Each stream contains
128 episodes and 3,072 query occurrences; global class 1 prevalence is 25%.
Occurrences are not necessarily independent accounts.

## Primary observation

| Stream / condition | Accuracy | Macro-F1 | Predicted class 1 | Corrected / corrupted versus intact | Within-episode AUC delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original / intact | 48.70% | 47.64% | 60.81% | — | — |
| Original / keys only | 48.34% | 47.52% | 62.53% | 101 / 112 | +0.15 pp |
| Original / values only | 25.91% | 21.45% | 98.83% | 248 / 948 | +6.86 pp |
| Original / removed | 25.16% | 20.24% | 99.84% | 239 / 962 | +8.76 pp |
| Fresh / intact | 49.67% | 48.27% | 58.53% | — | — |
| Fresh / keys only | 47.79% | 46.94% | 62.37% | 69 / 127 | +0.84 pp |
| Fresh / values only | 25.85% | 21.25% | 99.15% | 261 / 993 | +8.93 pp |
| Fresh / removed | 25.13% | 20.19% | 99.87% | 258 / 1,012 | +10.35 pp |

Joint K/V equals the removed condition in these exports. Value replacement
improves episode AUC in 83/128 original episodes and 71/128 fresh episodes;
it worsens AUC in 38 and 40, respectively (the remainder are tied). Thus the
mean ranking improvement is not confined to a single favorable episode, but
it is neither universal nor a classification improvement.

Fresh values-only confusion counts are TP=768, FP=2,278, TN=26, FN=0. The
intervention retrieves all minority-class positives while sacrificing almost
every majority-class negative. This is a poor deployed classifier despite the
improved ranking statistic.

## What the actual examples show

The following are the first query occurrences in the original stream satisfying
each outcome category, not the largest or most favorable effects. A positive
global margin predicts class 1. Episode 0 is the first stored batch; query indices
are zero-based in its returned query ordering.

| Episode / query | True global class | Native margin | Values-only margin | Outcome |
| --- | ---: | ---: | ---: | --- |
| 0 / 3 | 1 | -0.273114 | +0.180361 | Corrected |
| 0 / 7 | 0 | -0.074593 | +0.153367 | Corrupted |
| 0 / 0 | 1 | +0.088271 | +0.133424 | Remained correct |
| 0 / 6 | 0 | +0.842692 | +0.036325 | Remained wrong, much less confident |

The corrected and corrupted examples both cross the decision boundary in the
same direction. The last example shows why weaker confidence is not equivalent
to a corrected decision. These are actual prediction/activation cases, not an
inspection of biographies or account semantics; the private prediction export
does not itself include account identities or profile text.

## Geometry explains what not to infer

The geometric export defines reference contrast as
`||normalize(label_1) - normalize(label_0)||`. Its median falls from 0.25710 to
0.02251 on the original stream and from 0.26735 to 0.02178 on the fresh stream
under values-only replacement. The ratio of fresh medians is about 12.3; this
is not the median of episode-wise ratios.

Fresh native global margins have 10th/50th/90th percentiles
(-1.60257, 0.16820, 1.63367); values-only margins have
(0.04343, 0.10042, 0.18832). Margins become far more compressed and almost all
remain just above the same decision boundary. No exact-zero margins occur, so
the binary global-probability threshold and mapped local argmax agree here.

For cosine decoding, the margin is a positive scale times the dot product of
the normalized query with the difference of normalized class references.
Shrinking that difference alone cannot flip a margin's sign or improve its
within-episode ranking: its direction also matters. The intervention therefore
must not be described simply as repairing calibration or removing noise. It
changes the classifier direction and its contrast strength; higher AUC does not
show that the default decision boundary becomes suitable.

## Research decision

Retain this as mechanistic evidence and report classification metrics beside
AUC. Do not market context removal or value replacement as a practical repair.
Any subsequent repair must choose its operating point without target query
labels and beat the existing U1 readout baseline. Post-hoc target-optimal
thresholds would be oracle diagnostics, not deployable improvements.

This operating-point result does not invalidate the frozen values-over-keys AUC
prediction, but substantially limits its interpretation. The public multiclass
replication must retain accuracy, macro-F1 and NLL, not merely reproduce an AUC
ordering. This evidence is insufficient by itself for a method-led paper.

## Provenance and calculation

Tucker-only run root:
`/dataMeR1/phil/gfm/prodigy-classrefkv/log/classrefkv_long_20260907`.
Runtime revision: `5ffbfa2b3205ae8ac4d430daaa83060d6a92f34f`.
Read `private_predictions/{original,fresh}.pt`, `geometry.json`, and
`protocol.json`. Global labels are recovered by gathering `mapping` at `local_y`;
global-class-1 probability sums softmax columns whose mapping equals 1.
Accuracy/F1/confusion counts use this global alignment. Within-episode AUC is
computed separately by `episode_ids` and then averaged. Geometry is verified
against `run_class_reference_kv.py:geometry`, not inferred from field names.
