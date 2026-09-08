# Downstream evidence for the multi-graph pretraining paper

This audit assembles completed local result exports for slides 3 and 7 of
`SGFM Paper Presentation(1).pptx`. It launches no experiments and does not verify
live Tucker state. The evidence supports target-dependent benefits from adding
graphs; it does not establish a generally non-degrading pretraining framework.

## Coverage

| protocol | model | checkpoint_step | stream | cells | models | targets | training_seeds | minimum_sources | maximum_sources |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| prodigy_ladder | PRODIGY | 2500 | original | 375 | 25 | 5 | 3 | 1 | 9 |
| prodigy_lattice | PRODIGY | 2500 | original | 270 | 54 | 5 | 1 | 1 | 8 |
| prodigy_schedule | PRODIGY | 2500 | fresh | 135 | 27 | 5 | 3 | 2 | 4 |
| prodigy_schedule | PRODIGY | 2500 | original | 135 | 27 | 5 | 3 | 2 | 4 |
| samgpt_ladder | SAMGPT | 500 | original | 372 | 31 | 4 | 3 | 1 | 9 |
| vision_ladder | VISION | 2500 | original | 65 | 13 | 5 | 1 | 1 | 9 |

There are **1,352 protocol-specific physical model–seed–target–stream cells**.
The same checkpoint can occur in separate experiments; these are not 1,352
independent observations. Shared ladder endpoints are deduplicated in the cell
table and reused only in explicitly named order comparisons.

SAMGPT's native export uses different Facebook label tasks. Its five nonmatching
targets (three Facebook tasks, Cora, PubMed) are excluded here. The common panel
is COVID political, Election 2020, TwiBot-20, and Ukraine suspended. Other panels
also retain Facebook Page Reference. Even on the common panel, model-native
evaluation protocols and compute differ: compare within-model gains, not a
pooled cross-model score or an equal-compute architecture ranking.

## RQ2: all-nine versus each order's first specialist

The following AUC differences average training seeds first, then the four common
targets. Fraction improved and worst gain also use these seed-mean target gains.
Orders change which sources belong to each rung; they are not temporal training
schedules. All-nine has no unseen targets in this panel, so the unseen endpoint
metric is undefined, not zero. Consult earlier rungs in `data/deck_metrics.csv`
for genuinely unseen-target changes.

| model | order | mean_gain | fraction_improved | worst_gain | worst_target |
| --- | --- | --- | --- | --- | --- |
| PRODIGY | A | -0.0071 | 0.5000 | -0.0203 | ukr_rus_suspended |
| PRODIGY | B | 0.0591 | 1.0000 | 0.0053 | twibot20 |
| PRODIGY | C | -0.0009 | 0.7500 | -0.0262 | ukr_rus_suspended |
| SAMGPT | A | 0.0267 | 1.0000 | 0.0002 | ukr_rus_suspended |
| SAMGPT | B | -0.0212 | 0.2500 | -0.0514 | election2020 |
| SAMGPT | C | 0.0249 | 1.0000 | 0.0034 | ukr_rus_suspended |
| VISION | A | -0.0033 | 0.2500 | -0.0514 | covid_political |
| VISION | B | -0.0193 | 0.2500 | -0.0672 | covid_political |
| VISION | C | -0.0202 | 0.2500 | -0.0454 | covid_political |

The endpoint direction depends on the starting specialist. The same all-nine
checkpoint is reused across A/B/C within each seed; these endpoint comparisons
are not independent replications. VISION has one training seed and only odd
rungs. Its adjacent comparisons add two sources at a time, not one.

## RQ1: mixtures versus their best constituent specialist

These are wholly foreign-source comparisons: the target graph is absent from
the mixture. Best constituent is a per-target, per-metric oracle comparator,
not a deployable source selector. Each pair has two independently trained
specialists available as references; each leave-one-out mixture has eight.

| source_count | target_comparisons | mean_gain | fraction_improved | worst_gain | worst_target |
| --- | --- | --- | --- | --- | --- |
| 2 | 140 | -0.0108 | 0.2857 | -0.1570 | covid_political |
| 8 | 5 | -0.0093 | 0.0000 | -0.0148 | covid_political |

Here the fraction is over mixture–target comparisons (140 for pairs), not 140
distinct targets. `data/model_metrics.csv` supplies the fraction of targets
improved for each individual mixture. There is one training seed. Mixtures and
specialists have the same total updates but different per-source exposure;
negative gains do not identify causal interference.

## RQ1: exactly matched schedule comparisons

Fresh-stream AUC gains relative to interleaving, averaged over three training
seeds and then five targets. Blocked and replay100 use the same source-private
examples as their matched interleaved reference. Original-stream results remain
separate in the data. This table reports descriptive means and worst targets;
it does not replace the original study's crossed uncertainty analysis.

| comparison | source_count | mean_gain | fraction_improved | worst_gain | worst_target |
| --- | --- | --- | --- | --- | --- |
| blocked_versus_interleaved | 2 | -0.0010 | 0.2000 | -0.0139 | twibot20 |
| blocked_versus_interleaved | 3 | -0.0058 | 0.0000 | -0.0084 | ukr_rus_suspended |
| blocked_versus_interleaved | 4 | -0.0024 | 0.4000 | -0.0184 | twibot20 |
| replay100_versus_interleaved | 2 | 0.0030 | 0.6000 | -0.0184 | twibot20 |
| replay100_versus_interleaved | 3 | -0.0022 | 0.2000 | -0.0091 | covid_political |
| replay100_versus_interleaved | 4 | -0.0011 | 0.4000 | -0.0086 | ukr_rus_suspended |

## Definitions and interpretation

- `downstream_cells.csv`: outcomes, sources, target membership, training seed,
  checkpoint update budget, evaluation fingerprint, and input path.
- `target_performance.csv` and `performance_by_size.csv`: absolute AUC and
  accuracy, with seeds averaged before descriptive corpus-size summaries.
- `paired_gains.csv`: every seed-level paired metric and its exact reference.
- `target_gains.csv`: per-target seed means and observed seed ranges.
- `model_metrics.csv`: each mixture's mean gain, fraction of targets improved,
  worst gain, worst target, and in-mixture/unseen breakdown.
- `deck_metrics.csv`: order/rung or study-level aggregates, including common-four
  scopes, incumbent/newcomer roles, and adjacent-rung marginal gains.
- `coverage.csv` and `validation.json`: coverage checks and hashed inputs.

Positive gain means improvement. A win exceeds 1e-12, a numerical tie tolerance,
not a practical-effect threshold. Worst gain is the minimum seed-mean paired
gain; a negative value denotes degradation. Blank zero-target summaries are
intentional. Seed ranges are descriptive, not confidence intervals. Fixed
episode fingerprints do not create independent evaluation seeds or guarantee
identical sampled contexts across different studies. No between-protocol
contrasts are calculated.

The main results use AUC; accuracy is retained in all comparison exports.
Macro-F1 is not harmonized because it is not available consistently. Update
counts do not imply equal FLOPs, wall time, convergence, or per-source exposure.
Do not infer unseen performance from changing in-mixture/held-out panel means:
adjacent gains always pair the same target, then classify its new membership.

## Still missing from the deck

The table does not supply a downstream cross-graph sampling-ratio sweep, a
size-proportional-to-balanced exposure sweep, independent data-volume or
capacity sweeps, or a validated new pretraining framework. Earlier NM-only
controls cannot fill these downstream cells. GraphSAGE and adaptation-efficiency
results address different designs and are not spliced into these comparisons.
This is a classification evidence table, not evidence of transfer across every
downstream task; regression and valid pair-link prediction require separate
protocol-specific audits if the paper retains that broader claim.
