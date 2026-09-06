# Resampling the same supports is not the political-transfer repair

6 September 2026. Completed follow-up to natural support replacement. All six
Ukraine/Hong Kong checkpoints (three initialization seeds), five targets and both
128-episode streams are retained. There is no new training or additional label.

## Result and decision

Hong Kong remains more sensitive than Ukraine when the **same twenty labeled
accounts** are retained and only their sampled graph neighborhoods change. Its
political query-probability variance is 1.24–2.43 times Ukraine's, with the
prespecified direction holding in all six seed/stream comparisons.

But the effect is small as a performance remedy. Averaging eight neighborhood
draws changes Hong Kong political AUC by −.00121 to +.00062 relative to the
original draw. Its three-seed mean change is −.00007 on the original stream and
+.00009 on the second. It does not materially close the source gap. Do not spend
further inference compute on this as the proposed political-transfer fix.

| Political outcome across eight draws | Hong Kong | Ukraine |
|---|---:|---:|
| Fraction of queries ever switching correctness | 5.96–8.01% | 2.08–3.52% |
| Mean probability variance, range across seed/stream | .000909–.001419 | .000467–.000731 |
| Ensemble AUC change versus original, range | −.00121 to +.00062 | −.00035 to +.00014 |

The earlier intervention that changed support accounts had much larger
correctness-switching fractions (52–63% for Hong Kong, 13–18% for Ukraine).
The protocols differ, so this is not an additive variance decomposition or a
causal percentage attributed to account identity. It is enough to distinguish
the practical research directions: ordinary neighborhood-resampling noise is
not an adequate explanation or repair for the large observed political gap.
Which labeled accounts provide the evidence, and how the model processes that
evidence, remain more consequential leads. This does not imply graph context
is generally unimportant.

## Scope

Draw0 is the original context; draws1–7 use the original production sampler
around the same support centers. Queries, center features, support labels and
metagraph metadata remain unchanged. Every draw runs the full frozen model.
The primary question was directional sensitivity, not improvement from an
ensemble. Lower ensemble NLL than mean individual-draw NLL follows from Jensen's
inequality and is not counted as an empirical primary success.

Other targets remain in `fixed_support_context_effects.csv`. Hong Kong TwiBot
three-seed mean ensemble AUC gains are .00233/.00197; Facebook means reverse
sign across streams. Ukraine Suspension changes by −.00505/+.02299. These do
not establish a universal benefit from repeated sampling. The analysis concerns
query occurrences on already studied targets, not untouched-task generalization.
Greater probability variance can depend on model confidence and query difficulty;
it is not itself a causal estimate of the training-source effect.

## Evidence

The complete export contains 60 model/target/stream cells, 540 aggregate metrics,
61,440 episode/draw rows and 15,360 full-forward/suffix checks. Query inputs and
pre/post-metagraph query vectors are bit exact under support replacement.
The independent checker rehashed all 2,560 support batches and 320 original
batches and recomputed all 60 prediction cells; largest metric discrepancy is
1.71e-7 and variance discrepancy 1.73e-18. This work is complete, not a pending
experiment.

- Compact raw evidence: `data/fixed_support_context/`.
- Derived effects, sensitivity contrasts and cohorts: `data/fixed_support_context_*.csv`.
- Reproduce with the `analyze_fixed_support_context` analysis module.
- Runtime `973627ce`; completed output on Tucker at
  `/dataMeR1/phil/gfm/prodigy-mechanisms-fixedctx/log/fixedctx_eval_full_20260906`.
  Independent checker was run only after the original pipeline ended.
- Local branch `codex/target-performance-mechanisms`, worktree
  `/Users/philipp/projects/gfm/prodigy-mechanisms`.
