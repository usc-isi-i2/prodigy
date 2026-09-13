# Native versus foreign NM separation before and after the metagraph

9 September 2026. This audit reuses the same canonical HK and Ukraine episodes,
the frozen HK and Ukraine checkpoints, and the cached pre-metagraph and final
score matrices. It asks where native-only successes arise: candidate separation
before the metagraph, or conversion by the support-conditioned readout. No graph
loading, sampling, encoding, fitting, or GPU work was required.

## Method

For every query, compute the true-class score minus its best rival and divide by
the standard deviation of that query's 30 candidate scores. This makes the
cosine and final-logit margins comparable as within-episode standardized
separation. Ranks and correct/wrong transitions provide scale-free checks.
Models are paired on identical inputs and split into both-correct, native-only,
foreign-only, and both-wrong cohorts. HK is additionally split into uniquely
answerable and multiple-valid-answer cases.

All 245,760 pre-metagraph predictions and ranks reconstruct exactly. Final
scores reproduce all but two Ukraine-model predictions and one rank on HK;
the two winner gaps are at most 1.91e-6 and do not change the stored cohort
labels. All Ukraine-target and native-HK final rows reconstruct exactly.

## HK: native specialization is mostly readout conversion

On all 7,565 HK-native-only final successes, only 2,530 (33.4%) are already
correct under native HK pre-metagraph cosine; 853 (11.3%) are pre-metagraph
successes for the Ukraine model. The native HK model improves the true rank
from 4.71 to 1 by construction, a mean gain of 3.71 places, while the Ukraine
model worsens from 7.92 to 8.47 on the same inputs.

Ambiguous labels do not create this conclusion. Among 4,302 uniquely answerable
HK-native-only successes, 1,684 (39.1%) are correct before the metagraph and
2,618 (60.9%) are created by the final readout. Native HK mean rank moves
4.11→1.00; Ukraine moves 7.43→7.76. Native HK also has the better pre-metagraph
rank in 61.3% of this clean cohort, so it often starts with ordinal information
without placing truth first.

The readout helps outside successes as well. On 46,698 shared HK failures,
native HK mean rank improves 11.42→8.65 and standardized margin improves
−1.70→−0.76, whereas Ukraine mean rank changes 11.31→11.71. Most failures
remain failures because the HK correction is too small, not because the
metagraph uniformly destroys a good representation. Across all HK rows, native
HK rescues 6,139 pre-metagraph failures and loses 3,135 pre-metagraph successes;
Ukraine rescues 3,468 and loses 2,976.

## Ukraine: native specialization is mostly representation separation

On 18,881 Ukraine-native-only final successes, 13,130 (69.5%) are already
correct before the metagraph, compared with 2,834 (15.0%) for HK. Ukraine has
the better pre-metagraph rank in 80.4% of this cohort and a mean standardized
native-minus-foreign margin of +1.12. Its rank then improves modestly from
1.69→1.00, while HK worsens from 7.32→7.95.

On 31,407 shared Ukraine failures, Ukraine still begins ahead: mean rank 6.99
versus 10.98 for HK and a positive native-minus-foreign standardized margin.
The Ukraine readout improves mean rank to 6.41, but usually cannot finish the
decision. Across all Ukraine rows, the native model rescues 6,714 pre-metagraph
failures and loses 3,485 pre-metagraph successes; HK rescues 5,437 and loses
4,763.

## Interpretation

This sharpens the earlier aggregate stage accuracies. Source training does not
produce one universal failure location:

- Ukraine specialization mainly creates episode-relative query/support
  separation before the metagraph.
- HK specialization mainly converts partial ordinal signal through the learned
  support-to-class reference. Its readout is useful overall and also partially
  repairs shared hard cases, but starts from substantially weaker separation.
- Positive-support dominance is the common computation pathway, while the
  amount and stage of usable source-specific signal differ.

These are controlled-input computational facts, not a causal decomposition of
training. The cohorts are selected by final outcomes, so their margin sizes
describe how the models differ within those outcomes rather than predict unseen
success rates. Standardization removes row scale, but does not make separately
learned coordinates interchangeable.

## Consequence

The simple fixes tested so far target the wrong level. A global raw/pre-M blend
hurts native HK, contradicted-negative deletion hurts after training, and global
source-manifold affinity does not predict native success. The remaining
mechanism is episode-relative and source-dependent: some sources need better
pre-metagraph candidate separation, while HK needs a readout that can exploit
weak ordinal evidence without breaking existing successes.

Before designing that readout, replicate this exact decomposition on other
native graphs spanning strong donor, small-source, and island regimes. If the
stage balance tracks source properties, a learned confidence-gated residual
between pre-metagraph and metagraph scores becomes a principled candidate; it
must be trained on source episodes and evaluated on untouched multi-positive
and unique-anchor queries, with recovery and loss counts. The failed fixed
equal blend shows why an ungated version is insufficient.

Compact aggregate: [nm_source_margin_stages_20260909_v2.json](data/canonical_split/nm_source_margin_stages_20260909_v2.json).
Runtime branch `codex/nm-hk-goal-20260908`, revision `564240b3`; private log at
`/dataMeR1/phil/gfm/error_audit/nm_source_margin_stages_20260909_v2.log`. The
successful cached pass took under one minute on two low-priority CPU threads.
