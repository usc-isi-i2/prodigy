# Two-graph MLP error analysis

## Scope and protocol

Diagnostic cases: Ukraine/Russia + Facebook pages (negative transfer), and COVID political + Ukraine suspended (positive transfer). Each pair is evaluated on its six other targets among the eight non-Election graphs. These two target sets differ; this is a descriptive contrast, not a controlled causal comparison. Cases were chosen after seeing aggregate performance. Single seed, validation-selected checkpoints, not matched training compute.

Five saved models per target: each singleton, both sequential orders (AdamW preserved), and interleaved training from scratch. All 60 score arrays were checked for identical endpoint IDs, labels, and validation masks within each target comparison. Recomputed test AUC agrees with the saved reports. Each test cell has 1,400 positive and 7,000 negative pairs; validation has 600 positives and 3,000 negatives. No model retraining was performed.

For error accounting, the baseline singleton is chosen by target-validation AUC, not test AUC. Classification uses each model's saved threshold maximizing validation balanced accuracy. Calibration uses its saved nonnegative affine transformation fitted on validation only. Consequently these are diagnostics using target validation labels, not a label-free deployment procedure.

Ukraine is selected on all six negative-case targets. Ukraine suspended is selected on all six positive-case targets. On Midterm and TwiBot-20 the positive-case test-best singleton is instead COVID political. This explains the +3.34 pp diagnostic gain vs +3.11 pp over the retrospective per-target test-best baseline in earlier plots.

## Summary

| Pair/model | Test AUC | Delta vs validation-selected singleton (pp) | Raw BCE | Validation-calibrated BCE | Balanced accuracy |
|---|---:|---:|---:|---:|---:|
| Ukraine singleton | 0.95764 | 0 | 0.19884 | 0.17700 | 0.89881 |
| Ukraine→Facebook | 0.93622 | -2.14 | 0.62568 | 0.21394 | 0.86670 |
| Facebook→Ukraine | 0.95834 | +0.07 | 0.19627 | 0.17702 | 0.89840 |
| Ukraine+Facebook interleaved | 0.94152 | -1.61 | 0.21990 | 0.20513 | 0.87269 |
| Ukraine+Facebook ensemble | 0.94893 | -0.87 | 0.20349 | 0.20349 | 0.88212 |
| Suspended singleton | 0.86115 | 0 | 0.38059 | 0.30337 | 0.78769 |
| Political→Suspended | 0.87798 | +1.68 | 0.36440 | 0.28663 | 0.80596 |
| Suspended→Political | 0.88375 | +2.26 | 0.34823 | 0.27757 | 0.81807 |
| Political+Suspended interleaved | 0.89456 | +3.34 | 0.30517 | 0.27231 | 0.82512 |
| Political+Suspended ensemble | 0.88398 | +2.28 | 0.29154 | 0.29154 | 0.81493 |

Ensemble = unweighted mean of the two validation-calibrated singleton probabilities. No ensemble weights or test-dependent choices were fitted. Its threshold is also chosen on validation. This simple ensemble does not preserve the strong Ukraine singleton, whereas it captures some, but not all, of the positive pair's ranking improvement.

## Ranking changes are real

All positive-negative test comparisons were examined, including half credit for tied scores. Recovered minus damaged ordering fractions were verified to equal the AUC difference exactly (numerical tolerance).

- Ukraine+Facebook interleaving recovers 1.71% of all positive-negative orderings but damages 3.32%: net -1.61 pp AUC. It loses AUC on all six targets, from -0.67 pp on COVID-19 to -2.47 pp on Midterm.
- Political+Suspended interleaving recovers 8.35% and damages 5.01%: net +3.34 pp. It gains on all six targets against the validation-selected singleton, from +1.47 pp on China/HK to +7.28 pp on Midterm.

These ordering comparisons share endpoints and are not independent samples; no significance claims or binomial confidence intervals are made.

## Correct predictions lost and gained

Using validation-selected balanced-accuracy thresholds, rates below average equally across positive/negative classes and then across targets:

- Ukraine+Facebook: 5.80% previously correct predictions become wrong; 3.19% previously wrong become correct. Net balanced accuracy -2.61 pp. True edges lose 5.39% and gain 3.64%; non-edges lose 6.20% and gain 2.73%.
- Political+Suspended: 6.96% become wrong; 10.71% become correct. Net balanced accuracy +3.74 pp. True edges lose 6.93% and gain 10.57%; non-edges lose 7.00% and gain 10.84%.

Thus synergy is not error-free accumulation: both cases lose some useful predictions. The positive pair recovers substantially more than it destroys.

## Confidence and decoder effects

Ukraine→Facebook has a particularly severe confidence failure: negative BCE rises from 0.0617 to 0.7079, while positive BCE improves from 0.8847 to 0.2144. It becomes too willing to call non-edges positive. Calibration reduces total BCE from 0.6257 to 0.2139, but that remains worse than calibrated Ukraine (0.1770), and AUC remains lower. A scalar bias cannot change within-target ranking.

For Ukraine+Facebook interleaving, positive BCE improves slightly (0.8847→0.8416), but negative BCE worsens (0.0617→0.0956). Calibrated BCE remains worse (0.1770→0.2051). For Political+Suspended interleaving, positive BCE improves sharply (1.9663→1.3424); negative BCE worsens (0.0635→0.0977), with a net improvement in total and calibrated BCE.

## Where errors concentrate

Graph covariates were computed in place on Tucker. Each evaluation pair has endpoint feature cosine, mean feature norm, neighbor-mean cosine, minimum/maximum context degree, and minimum available sampled neighbors. Context-only degrees were used in the reported analysis. Full-known-edge degrees were extracted only as descriptive metadata, not used for decisions. Feature quartile boundaries come from the validation subset, separately for each target and class.

Ranking credit for a positive edge is its fraction of negatives outranked; for a negative edge it is the fraction of positives above it. Changes are descriptive contributions relative to the entire target's opposite class, not AUC recomputed on an isolated subgroup.

**Ukraine+Facebook**

- True edges in the lowest node-feature-similarity quartile lose 2.60 pp ranking credit; highest quartile loses only 0.41 pp.
- High-similarity non-edges lose 2.70 pp, versus 0.89 pp for low-similarity non-edges. False-positive damage is therefore concentrated among non-edges whose endpoint features look similar.
- True edges with at least one endpoint lacking sampled context lose 3.97 pp ranking credit; edges where both endpoints have ten sampled neighbors lose only 0.12 pp. All six targets contribute to both groups, but these groups differ in degree and difficulty, so this is not a causal isolation of neighbor effects.
- Feature-norm quartiles have similar effects, roughly -1.5 to -1.9 pp for positives. Input norm alone does not isolate the failure.

**Political+Suspended**

- Low-similarity true edges gain 4.94 pp ranking credit; high-similarity true edges gain 1.66 pp.
- High-similarity non-edges gain 4.27 pp; low-similarity non-edges gain 2.85 pp.
- Gains appear across context-coverage bins, including +5.35 pp for true edges with a minimum of 1–4 sampled neighbors, and +2.48 pp with ten at both endpoints.

The two cases move the difficult low-similarity positives and high-similarity negatives in opposite directions. This is consistent with different learned uses of feature similarity and context; it does not establish an embedding-preprocessing cause. Neighbor/degree quartiles can collapse because of ties at zero and therefore sometimes contain fewer than six targets; use `targets` and `n` in aggregate_strata.csv when interpreting them. The node-cosine quartiles include all six targets.

## Direction versus embedding magnitude

For dot scores, z_u·z_v = cosine(z_u,z_v) × ||z_u||||z_v||. Saved dot and cosine scores allow a descriptive swap of pairwise magnitude products and cosine components between the baseline and interleaved model. All test pairs had nonzero-enough cosine for reconstruction; none were dropped.

| Score construction | Ukraine+Facebook AUC | Political+Suspended AUC |
|---|---:|---:|
| Baseline dot | .95764 | .86115 |
| Interleaved dot | .94152 | .89456 |
| Baseline magnitudes × interleaved cosine | .94468 | .88376 |
| Interleaved magnitudes × baseline cosine | .95261 | .87942 |
| Baseline cosine alone | .90769 | .84199 |
| Interleaved cosine alone | .90956 | .86642 |

Keeping baseline magnitudes alone recovers only 0.32 pp of the negative pair's 1.61 pp loss. Keeping baseline cosine with interleaved magnitudes recovers 1.11 pp but still loses 0.50 pp. Neither component explains the full behavior independently. Cosine normalization alone is much worse than the strong singleton's dot score, so blindly normalizing embeddings is not supported by this result. These swaps combine two separately trained models; they are diagnostics, not a proposed deployable encoder.

## Next experiment

Do not start by merely merging graphs or changing the scalar bias. A useful next test is an interleaved model initialized from the strong singleton, with a teacher constraint on its pair scores/rankings over source-A training pairs while learning B. Compare to identical warm-start interleaving without that constraint, using source validation only for selection and keeping these downstream tests for final evaluation. This directly tests whether protecting the strong model's ranking function reduces the observed destruction; it is a proposal, not established effectiveness. Keep the singleton available as a validation-selected fallback.

For now the evidence supports two different regimes: genuine complementary ranking gains for Political+Suspended, and a harmful compromise for Ukraine+Facebook. One universal equal-weight mixture rule is not uniformly beneficial.

## Files and provenance

- Local artifacts: `/tmp/mlp-error-analysis/` (`metrics.csv`, `transitions.csv`, `selectors.csv`, `aggregate_strata.csv`, `norm_swap.csv`, diagnostic PNG/SVG).
- Source code branch: `codex/mlp-pair-error-analysis`; local worktree `/tmp/mlp-pair-error-code`; commits `4a99d53`, `c1c47e0`.
- Tucker analysis worktree: `/dataMeR1/phil/gfm/mixture-scaling-pair-errors`.
- Tucker covariates and per-edge diagnostics remain under `/dataMeR1/phil/gfm/mixture-scaling/results/pair_errors_20260912/`.
- Automatic approval review blocked local export of per-node covariates. Analysis ran in place instead; only aggregate covariate statistics were returned.
