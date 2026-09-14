# ogbl-collab: findings from the continuation and error-case discussion

Written September 14, 2026. Objective: exceed HyperFusion's reported 71.29%
Hits@50 with fewer than one million learned or fitted inference scalars under an
accepted evaluation protocol. **That objective remains unmet.**

## Evidence status and current result

The historical best fresh-negative standalone test seed remains **70.1569%**.
The fresh validation-selected fused system averaged **69.7979% ± 0.1786 percentage
points** across three optimization seeds, with 496,448 inference scalars. The best
single seed is not an estimate of expected performance. Seed dispersion is not an
independent-data confidence interval.

Earlier 80%+ results are not clean wins: one campaign exposed 99,984 of 100,000
official test negatives through reused validation negatives; other diagnostics
used test supervision or test-dependent selection. Prior extensive 2019 exploration
must remain disclosed. None of the follow-ups summarized here establishes
leaderboard acceptance, and their recorded protocols report no new 2019 scoring.

## Completed experiment and diagnostic findings

### Two-model averaging failed its validation gate

Equal-logit averaging of the three possible fresh-expert pairs gave 68.5607%,
68.5490%, and 67.8933% on 2018 validation. The best pair, seeds 0+1, underperformed
the validation-selected standalone control at 68.7155%, losing 93 net hits.
Conservative pair inference budget: 992,897 scalars. The frozen gate stopped the
experiment without a new test evaluation.

For seeds 0+1, averaging recovered 612 control misses but lost 705 control hits.
The models shared 35 of their top 50 negatives, with negative-score Spearman
correlation 0.913. Their average slightly improved AUC (0.977046 to 0.977470) while
reducing Hits@50. Thus broad ranking improvement and seed diversity were insufficient
at the extreme-negative boundary.

This experiment reused fusion-selected checkpoints. Standalone validation peaks
for seeds 1 and 2 occurred at update 100 rather than the selected updates 200 and
250. Consequently, it rejects the specified averaging recipe at those checkpoints,
not every possible ensemble. Subsequent matched tests used standalone selection
for both controls and treatments.

### Negative-panel overfitting is directly observed

“Fresh negatives” originally meant one fresh 100,000-pair pool per year, repeatedly
mined during training. A saved-checkpoint probe held the year, graph, and 119,622
positive pairs fixed while replacing the training negatives with a disjoint
100,000-pair historical panel. At update 2,000:

| Seed | Training negatives | Separate same-year negatives |
|---|---:|---:|
| 0 | 99.92% | 64.82% |
| 1 | 99.91% | 68.52% |
| 2 | 99.91% | 67.42% |

For seed 0 the 50th-negative logit was −2.89 on training negatives versus +18.74
on separate negatives. Negative-generalization failure therefore occurs without
a year change. Positives deliberately remain training examples in this probe;
these are diagnostic scores, not benchmark generalization estimates. All twelve
seen/unseen aggregate metrics were independently reproduced from saved scores.

### Fixing late overfit did not improve selected performance

Completed follow-up artifacts in the workspace supersede the earlier proposal to
try renewal and tail loss. Twelve matched training cells used three seeds per arm,
2,000 updates, and identical standalone validation selection.

| Recipe | Mean selected 2018 Hits@50 | Difference from control |
|---|---:|---:|
| Fixed-pool BCE control | 68.8908% | — |
| Renew negative pool every 50 updates | 68.8164% | −0.0743 pp |
| BCE plus fixed extreme-tail ranking loss | 68.2467% | −0.6441 pp |

At update 2,000, respective means were 55.6615%, 62.7239%, and 66.9807%.
Both interventions materially improved late performance while failing to improve
the best selected checkpoint. **The overfitting diagnosis is supported; the claim
that these particular remedies raise the best attainable score is not.**

The temporal path-age prerequisite also stopped before feature training. Younger
paths were directionally associated with positives in 2017 and 2018, but only
25 and 27 usable negative groups remained, below the frozen minimum of 30.
After deleting the best 20% of groups, preference was only 52.33% and 51.27%.
This is insufficient support, not proof that temporal path features cannot help.

## Hits@50: the important error is a missed positive

For a positive score s(p), the audited evaluator counts a hit when
**s(p) > the 50th-largest negative score**. Hits@50 is the fraction of positive
examples satisfying that condition. “False negative” here is shorthand for a
positive below this ranking boundary, not an error at a fixed probability cutoff.

Negative scores matter through the shared boundary. Reducing a highly ranked
negative while it remains above the boundary may have no effect at all. A useful
intervention must either raise missed positives above the boundary or lower the
boundary enough to recover them. The diagnostic goal is:

**Net recovered positives = newly recovered positives − previously correct
positives lost, after recomputing the modified model's negative cutoff.**

AUC, training accuracy, lower negative scores, and the number of rescued positives
alone are insufficient success criteria. In particular, suppressing the single
highest-scoring negative is not automatically valuable.

## Where the misses occur

On 2018 validation, fresh seed 0 correctly ranks 99.9679% of 28,072 repeated
collaborations but only 41.3095% of 32,012 novel collaborations. Of 17,794 positives
missed by all three seeds, 17,785 are novel; only nine are repeats. Repeat-only
improvements have almost no remaining headroom on this panel.

The same model improves zero-AA positive recall from AA-DC's 9.16% to 20.68%,
but lowers nonzero-AA positive recall from 98.86% to 94.72%. Relative to AA-DC,
it recovers 2,793 positives while losing 1,976. This motivates studying preservation
of useful structural evidence alongside new-collaboration recall, rather than
unconditionally boosting similarity or activity.

These strata overlap; their counts must not be added as disjoint error categories.
Their validation behavior is not a measured prediction of the 2019 distribution.

## Concrete error examples

Source: fresh seed 0, update 150, official 2018 validation. The logit cutoff was
0.615729928; scores are not probabilities. Numbers identify dataset nodes, not
verified author names. Examples were deliberately selected to illustrate distinct
regimes, not sampled to estimate their prevalence.

| Pair | Benchmark outcome | Logit | Observed evidence |
|---|---|---:|---|
| 179010–110316 | Positive missed | −0.2693 | Three prior collaborators per endpoint, no 2017 activity, feature cosine 0.865; common-neighbor evidence. AA-DC correctly ranks it. |
| 735–99694 | Positive missed | −0.0856 | Cosine 0.966, both recently active, nonzero length-three signal, no common neighbor; first observed collaboration. |
| 199950–209860 | Positive missed | 0.0136 | Prior collaboration in 2016, cosine 0.631, one endpoint inactive in 2017. AA-DC correctly ranks it. |
| 67708–70457 | Official negative ranked highly | 2.4514 | Both recently active, cosine 0.945, common-neighbor evidence. AA-DC also ranks it too highly. |
| 58387–61963 | Highest-scoring official negative | 7.8237 | Collaborated in 2017, both active, cosine 0.955. Strong recurrence evidence misleads both scorers for the target year. |

An official negative is a negative for this benchmark target, not a claim that the
pair never collaborated. For novel pairs, the feature value last_event_age=20 is
a capped/default encoding and must not be described as an actual event twenty years
ago. Recent-activity counts are graph-derived event incidences, not verified paper
counts. Similarity and activity are associations here; no causal feature attribution
was performed.

The cases show two distinct possibilities: the neural model can discard structural
signals that would have recovered a true positive, and plausible structural or
recency signals can still describe target-year negatives. Neither “trust AA always”
nor “boost active similar pairs” follows from these examples.

## Untested proposal from the recall discussion

A remaining proposal is a **compact AA-anchored residual scorer** trained to recover
near-boundary positives while retaining reliable existing hits. An operational
version would use a fixed training-derived AA score transformation plus a bounded
learned correction, and compare against the current joint scorer with matched
sampling, budgets, and standalone selection. No implementation or result for this
specific proposal is claimed here.

Important qualifications:

- Earlier global AA fusion and shallow learned gates failed. This would need to
  establish a benefit from the residual parameterization or explicit hit-retention
  objective, not rename one of those failed approaches.
- A bounded or nonnegative correction does not guarantee preservation of hits:
  corrections to negatives may raise the evaluation cutoff.
- Negative renewal and the tested tail loss must not be advertised as demonstrated
  improvements. If used, they belong in both matched arms or require a new explicit
  hypothesis; combining them is not automatically supported by their diagnostics.
- Assess full-panel net hits, with AA-supported versus zero-AA and novel versus
  repeated strata reported separately. Train against training data only; use
  chronological validation to select. Do not optimize individual official negatives.
- Before another campaign, specify which untested mechanism distinguishes the new
  objective from the already-failed BCE-plus-tail-loss recipe, and define a stopping
  rule. No further training, refit, test evaluation, or submission is authorized by
  this write-up.

## Evidence and scope

This note consolidates this conversation and the completed follow-up records found
in the shared workspace. The latter were read from the existing integrated evidence
at revision 2d80d968; their runs were not repeated for this write-up. No new model
execution or test access was needed to prepare it.

Relevant analysis leaves under scripts/experiments/analysis/baselines/:

- ogbl_collab_fresh_ensemble: frozen averaging experiment and audit.
- ogbl_collab_fresh_diagnosis: dynamics, fixed-year probe, independent metric audit.
- ogbl_collab_parallel_hypotheses: completed follow-up comparisons and audits.
- ogbl_collab_recall_cases: this note and exact illustrative pair records.

The copied error-example records came from score and panel archives whose hashes
matched the admitted validation diagnosis. The canonical compact-joint consolidated
findings retain the historical results and earlier invalid-result disclosures.

## Uniform sample follow-up

A first-draw uniform sample of 20 of the baseline's 18,797 validation misses
finds 16 without a common collaborator, 4 recovered by frozen AA-DC,
and 8 scoring below more than 1,000 negatives. These cases temper the earlier
hand-selected emphasis on structural retention and near-boundary errors.
See [all 20 cases](UNIFORM_20_MISSES.md) and [sampling evidence](data/uniform_20_misses.json).
No new model training or test access.

## Second uniform sample

Twenty more misses were drawn uniformly from the remaining 18,777, excluding
the first sample (PCG64 seed 20260915). Again 16 lack a common collaborator
and 8 score below more than 1,000 negatives. Frozen AA recovers two; the
2018-calibrated reference recovers one additional case, disclosed separately.
See [second set of twenty](UNIFORM_20_MISSES_SECOND.md). No test access.
