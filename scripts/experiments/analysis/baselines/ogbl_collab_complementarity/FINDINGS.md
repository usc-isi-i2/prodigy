# AA-DC and MLP validation complementarity

## Follow-up: zero-self scoring

The user-authorized score intervention confirms the self-pair mechanism. Setting
the final combined score to zero whenever both endpoints are identical improves
the best common-weight validation result from **67.6814% to 67.8250%**, versus
**67.3557%** for AA-DC. The gain over AA-DC is now **0.4693 percentage points**,
up from 0.3257 points. The best common weight remains alpha = 0.5.

The same 60,084 positive and 100,000 negative pairs remain in evaluation. Only the
one negative self-pair's final score changes. Every non-self score, original
normalization constant, checkpoint, and alpha is held fixed. The final combined
50th-negative threshold is recomputed. The rule is based on endpoint identity,
not labels, and applies to positive and negative pairs alike; this validation
panel has no positive self-pairs.

| Alpha | Original mean Hits@50 (%) | Zero-self mean Hits@50 (%) | Zero-self gain over AA-DC (points) |
| --- | ---: | ---: | ---: |
| 0 | 67.3557 | 67.3557 | 0.0000 |
| 0.10 | 66.8104 | 67.4999 | +0.1442 |
| 0.25 | 66.9829 | 66.9995 | -0.3562 |
| 0.50 | 67.6814 | 67.8250 | +0.4693 |
| 0.75 | 67.5094 | 67.7191 | +0.3634 |
| 1.00 | 67.4306 | 67.6386 | +0.2829 |
| 1.50 | 66.7088 | 66.7643 | -0.5914 |
| 2.00 | 66.3205 | 66.3527 | -1.0030 |

At alpha = 0.1, zero-self scoring restores the normalized negative threshold from
1.1865 to exactly 1 in every seed. No new negative crosses the old threshold.
The combination recovers 113, 81 and 66 positives for seeds 0, 1 and 2,
respectively, with **zero lost AA-DC hits**. This isolates the self-pair as the
cause of the original low-weight penalty.

At alpha = 0.5, the corrected results are:

| Seed | Corrected Hits@50 (%) | Gain over AA-DC (points) | Recovered positives | Lost positives |
| --- | ---: | ---: | ---: | ---: |
| 0 | 68.0581 | +0.7024 | 959 | 537 |
| 1 | 67.6336 | +0.2779 | 731 | 564 |
| 2 | 67.7834 | +0.4277 | 759 | 502 |

The self-pair is not the whole limitation: at that weight 9--13 additional
non-self negatives still cross AA-DC's old threshold, and hundreds of positives
remain lost as a result. Larger rescue weights continue to underperform AA-DC.

For the standalone MLP, applying the rule to its positive exponential score
changes validation Hits@50 only from 44.8572 to 44.8905 (seed 0), 42.8300 to
42.8317 (seed 1), and 43.4009 to 43.4042 (seed 2). It does not explain the large
standalone MLP versus AA-DC performance gap.

This is a local NumPy intervention on the hash-verified archive; no training,
inference, graph computation, or test scoring was run. All 24 original cells were
reproduced, all 24 corrected cells completed, and equality of all non-self scores
was asserted. The executable analysis and source hash are preserved in
[self_pair_check.py](self_pair_check.py) and [self_pair_check.json](data/self_pair_check.json).
These results remain post-hoc validation evidence on a panel previously used for
selection. They do not establish a test-set improvement.

## Finding

The saved nonlinear feature MLP recovers some validation positives missed by
AA-DC, but it also promotes different hard negatives. A fixed, simple rescue
curve gives only a modest net improvement: the best common weight among the
predeclared candidates raises validation Hits@50 from **67.3557% to 67.6814%**
averaged across the three MLP training seeds (**+0.3257 percentage points**).
This is exploratory validation evidence, not a new test result or proof that
we beat AA-DC's 68.02% test score.

## Paired positive and negative diagnostics

All models score the same 60,084 official validation positives and 100,000 shared
negatives. AA-DC misses 19,614 positives. Each model's hit is defined by its own
50th-highest negative score, using the official strict-greater-than rule.

| MLP seed | MLP validation Hits@50 (%) | AA-DC misses recovered by MLP | Share of AA-DC misses recovered | Shared identities among each model's top 50 negatives |
| --- | ---: | ---: | ---: | ---: |
| 0 | 44.8572 | 2,529 | 12.89% | 5 |
| 1 | 42.8300 | 2,171 | 11.07% | 4 |
| 2 | 43.4009 | 2,318 | 11.82% | 5 |

About 95--96% of the recovered positives have raw AA score zero. This localizes the
complementarity to the region where AA-DC has weak structural evidence. It does
not establish that these are all novel collaborations: zero-AA and recurrence are
different partitions.

The union of separate hit masks covers 70.97--71.56% of positives, but that is a
label-aware complementarity ceiling for an OR of the separate hit decisions. It
is not the Hits@50 of a combined scorer: the two scorers bring substantially
different negative tails.

## Combined scoring with the negative threshold recomputed

The fixed diagnostic score is
`max(AA_DC / AA_DC_negative50, alpha * exp(MLP_logit - MLP_negative50))`.
Its negative threshold is recomputed for every seed and alpha; positives are not
counted against the original threshold after adding the feature score.

| Alpha | Mean validation Hits@50 (%) | Mean change from AA-DC (points) |
| --- | ---: | ---: |
| 0 | 67.3557 | 0.0000 |
| 0.10 | 66.8104 | -0.5453 |
| 0.25 | 66.9829 | -0.3728 |
| 0.50 | 67.6814 | +0.3257 |
| 0.75 | 67.5094 | +0.1537 |
| 1.00 | 67.4306 | +0.0749 |
| 1.50 | 66.7088 | -0.6469 |
| 2.00 | 66.3205 | -1.0352 |

At the best common setting, alpha = 0.5:

| MLP seed | Combined Hits@50 (%) | Recovered positives | Lost positives | Net positives | New negatives in top 50 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 67.7252 | 896 | 674 | +222 | 8 |
| 1 | 67.5454 | 710 | 596 | +114 | 8 |
| 2 | 67.7735 | 757 | 506 | +251 | 6 |

All three seeds improve at that setting, but most of the rescued positives are
offset by other positives falling below the higher negative threshold. The three
seeds share one validation panel; they are not three independent data replications.

## One concrete source of negative promotion

The official validation panel contains one negative self-pair (row 36,270,
zero-based), and no positive self-pairs. AA-DC explicitly assigns it score zero.
A nonzero embedding compared with itself has cosine one, so the saved MLP assigns
it a high score in every seed.

At alpha = 0.1, this is the **only new negative** that crosses the old normalized
threshold for every seed. Its promotion raises the 50th-negative threshold from
1 to 1.1865. The combined models gain only 46--89 positives while losing 389--394.
This explains the immediate decrease at low rescue weights. Other hard negatives
enter as the weight grows; the self-pair alone does not explain the whole curve.

No pair was removed or rescored for this diagnostic. An explicit self-pair scoring
rule would be a separate model change and should retain every official negative.

## Evidence and reproduction

- Producing revision: `d0c03e26d708b25aa048834b6a370af3cdfb8064`.
- AA-DC upstream: `b499c2046cfe76448545dfe08fad9effb58dd076`; source bytes verified
  against that revision. Validation replay gives 0.6735570201717596, matching the
  recorded 0.673557 to its printed precision.
- Original nonlinear MLP checkpoints from `official_v1`, seeds 0--2, epochs 137,
  109 and 103. All checkpoint and frozen-selection hashes passed. Every MLP replay
  matches its saved validation metric exactly.
- All three seeds and all 24 fixed rescue-curve cells completed. A separate local
  audit verified the score archive hash and recomputed every cell's hit, recovery
  and loss counts.
- Dataset and validation-pair fingerprints, exact paths, and checkpoint hashes are
  in the receipt below. Official negative arrays, including self-pairs, are intact.
- Runtime: 49.47 seconds, CPU with eight Torch threads. W&B recorded offline.
- Local worktree: `/Users/philipp/projects/gfm/prodigy-collab-complementarity`.
- Tucker worktree: `/dataMeR1/phil/gfm/prodigy-collab-complementarity`.
- Branch: `codex/collab-validation-complementarity`.
- Runtime scores and W&B: `/dataMeR1/phil/gfm/ogbl_collab_complementarity/validation_v1/`
  on Tucker. No test scores were computed. The existing dataset audit reads split
  arrays for fingerprint verification, including the test split, but test arrays
  are discarded before scoring.

Both AA-DC calibration and MLP checkpoint selection already used this validation
panel. The best rescue weight is also identified on this panel. Therefore this
check supports complementarity and identifies a concrete scoring mismatch; it does
not estimate an independently validated improvement or justify a test-score claim.

Sources: [setup and fixed diagnostic contract](../../../setup/ogbl_collab_complementarity/README.md),
[full results](data/results.json), [replay receipt](data/validation_receipt.json),
[score audit](data/score_audit.json), and [archive audit script](audit_scores.py).
