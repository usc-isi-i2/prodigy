# Canonical NM: what distinguishes failures from successes?

8 September 2026. Native Ukraine and Hong Kong models only. Read-only analysis of
the saved corrected held-out NM predictions and completed support intervention;
no new model forwards. Each target contains 512 episodes, 30 anchor classes per
episode, four queries per class, and 61,440 query occurrences.

## Main conclusion

Failures concentrate on high-degree, frequently repeated query nodes and on
entire anchor classes. Raw query-to-support text similarity alone is a weak
separator. Changing supports establishes conditional support dependence for some
cases, but the failed cases rescued by that intervention skew toward lower
degree. Support robustness training is therefore motivated, without evidence
that it will solve the persistent high-degree failures.

An episode is not one binary prediction: it contains 120 queries. There are no
fully correct or fully incorrect episodes here. Episode accuracy ranges from
24.17–64.17% on Ukraine and 7.50–30.00% on Hong Kong. Below, query outcomes,
anchor-class outcomes, and whole-episode difficulty are kept separate.

## Query-level differences

| Descriptor | Ukraine incorrect | Ukraine correct | Hong Kong incorrect | Hong Kong correct |
|---|---:|---:|---:|---:|
| Query occurrences | 34,317 | 27,123 | 50,464 | 10,976 |
| Median full-graph incident degree | 3,383 | 483 | 1,486 | 654 |
| Median occurrences of that query in test | 3 | 1 | 110 | 34 |
| Same query assigned multiple anchors within episode | 3.66% | 1.05% | 24.95% | 14.70% |
| Zero query feature vector | 5.91% | 9.21% | 7.24% | 9.78% |
| Mean raw query–true-support cosine | .5610 | .5597 | .5829 | .5855 |

These are occurrence-weighted descriptive associations. Degree, frequency and
ambiguity are correlated; they are not three independently identified causes.
Degree is incident-edge count from the original full graph, not sampled-context
size or training-only degree. Zero features are actually less frequent among
errors here, so missing query text alone is not an adequate explanation.

Support cosine averages the three original GTE query–support similarities and
requires all four vectors to be nonzero. Valid counts are 28,766/20,477 for
Ukraine wrong/correct and 37,552/7,673 for HK; other rows are missing, not zero.
These are raw features, not the learned encoder or metagraph representations.

Using validation-defined support-cosine quintile boundaries, test accuracy spans
40.35–43.42% on Ukraine and 16.35–18.08% on HK, with no strong common monotonic
pattern. Within queries observed both correct and incorrect, the mean support
cosine is .0132 higher when correct on Ukraine (2,885 eligible queries) and .0092
higher on HK (1,018). This modest association changes anchor/candidate conditions
too; it is not a causal support-quality effect. Query+anchor matching offers
little support-feature variation under the historical selection policy and
cannot establish that feature similarity is irrelevant.

## What wins when the model is wrong?

| Error descriptor | Ukraine | Hong Kong |
|---|---:|---:|
| Correct anchor is rank 2 or 3 | 43.18% | 25.48% |
| Correct anchor is below rank 10 | 17.30% | 29.55% |
| Median correct-anchor rank among errors | 4 | 6 |
| Wrong predicted class has more text-similar supports than true class | 59.37% | 58.10% |

The last row includes 26,022/31,982 errors with valid true and predicted support
cosines. Mean true-minus-predicted support cosine is −.0135/−.0146. Conversely,
the true supports are more similar in 40.63%/41.88% of valid errors. Wrong-anchor
text itself is more similar in only 48.29%/47.96% of valid anchor comparisons.
Thus some mistakes resemble support-level semantic competition, but raw text
nearest-neighbor confusion cannot explain all errors. Comparing only the
model-selected wrong class is outcome-conditioned; this is not a causal test or
an evaluation of a nearest-prototype classifier over every candidate.

## Anchor-class failures and whole episodes

| Class outcome | Ukraine all four wrong | Ukraine all four correct | HK all four wrong | HK all four correct |
|---|---:|---:|---:|---:|
| Classes out of 15,360 | 4,063 | 2,600 | 8,843 | 369 |
| Median query degree within those classes | 5,183 | 289 | 1,636 | 11 |
| Multi-anchor query occurrence fraction | 4.09% | 0.38% | 26.63% | 5.89% |

The earlier within-query permutation already found excess all-four-wrong classes
beyond the observed query difficulty mix; see [difficulty audit](FINDINGS_NM_DIFFICULTY.md).
The table above describes those classes, without identifying whether their
anchors, supports, competing classes or sampled contexts caused the joint errors.

For a whole-episode comparison, take the bottom and top 103 episodes by native
accuracy, with stable input-order tie breaking (approximately quintiles):

| Descriptor | Ukraine bottom / top | Hong Kong bottom / top |
|---|---:|---:|
| Average episode accuracy | 34.92% / 53.11% | 12.52% / 23.54% |
| All-four-wrong class fraction | 35.83% / 18.06% | 65.86% / 49.06% |
| Mean fraction of predictions going to most selected anchor | 9.72% / 8.45% | 12.42% / 11.55% |
| Mean number of different anchors predicted, out of 30 | 28.02 / 28.42 | 24.63 / 25.15 |

Worse episodes show somewhat more concentrated predictions, but not wholesale
collapse onto one anchor. Their average support cosine is essentially unchanged
(.5611/.5578 Ukraine and .5839/.5820 HK). The accuracy contrasts themselves are
by construction, not evidence of a newly discovered episode type.

The aggregate also records a validation-only query-difficulty expectation
(per-query validation accuracy shrunk by ten observations toward the validation
mean, unseen queries assigned that mean). Its episode rank correlation with
test accuracy is .465 Ukraine/.263 HK. This is a descriptive, uncalibrated
predictor, not variance explained; its shrinkage and split shift preclude a
causal decomposition of the bottom/top accuracy gap.

## Which failed cases respond to new supports?

Among the 100 initially failed, single-anchor cases per target:

| Descriptor | Ukraine rescued at least once / never | Hong Kong rescued at least once / never |
|---|---:|---:|
| Cases | 38 / 62 | 27 / 73 |
| Median query degree | 522 / 8,361 | 65 / 914 |
| Median original true-anchor rank | 2 / 5 | 5 / 6 |

This is a post-hoc split of the existing five-draw intervention. "Never" means
not rescued in those five draws, not unsolvable. Degree/frequency matching was
between the original failed and correct cohorts, NOT between rescued and
unrescued failures. Pool size and other properties remain uncontrolled here.
HK's rescued mean baseline rank is 8.44 versus 8.03 for unrescued cases, so the
near-miss pattern is not consistent across targets. Lower degree is the clearer
shared descriptive pattern.

## Evidence and reproduction

[Analysis script](audit_nm_episode_detail.py) ·
[Aggregate results](data/canonical_split/nm_episode_detail.json).

Inputs were retrieved read-only from Tucker to `/private/tmp/nm_episode_detail`:
the two `paired_cluster_queries_private.tsv` tables beneath
`/dataMeR1/phil/gfm/error_audit/nm_canonical_split_bios_20260908/`, plus
`results_private.csv` and the two case manifests beneath
`/dataMeR1/phil/gfm/error_audit/nm_support_resampling_20260908_v2/`.
Only aggregate output is placed in the repository; no private node-level tables
or bios are included. Input hashes are recorded. The script asserts row
uniqueness, episode/class counts, and five draws per case/condition.

Run locally with Homebrew Python 3.11 and `--input-root` pointing to those five
files (named `ukr.tsv`, `hk.tsv`, `draws.csv`, `ukr_cases.json`, `hk_cases.json`).
Analysis performed in `/Users/philipp/projects/gfm/prodigy`, branch `main`.
One checkpoint seed per target, exploratory comparisons, no retraining or causal
claim about source training. No new model inference was necessary.
