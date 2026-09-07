# Support-calibrated repair exchanges errors rather than preserving ridge decisions

7 September 2026. Private saved-prediction analysis, zero new model forwards.
This incorporates a completed test from `codex/support-boundary-repair` without
changing that branch, its protocol, or its conclusions.

## Verified experiment

Read Tucker `prodigy-boundary-repair/log/boundary_full_20260907/{DONE,protocol,metrics}.json`.
The run has 128 episodes, six conditions, 3,072 query occurrences, and is not
a smoke run. It uses the nominated HK 50k checkpoint, political target, new
episode offset 200009, three supports per class. Calibration uses balanced
support-held-out folds and fixed penalized slope/intercept fitting; slopes
are unconstrained. Ridge uses normalized pre-metagraph U1 features, lambda=1,
one-hot support targets, no intercept. No query labels enter either fit.

Plain ridge accuracy is .597005 versus .588216 for calibrated value replacement;
macro-F1 .556920 versus .522853; pooled AUC .623944 versus .550258. The tested
repair does not beat the stronger readout. This does not exclude all possible
calibration methods or establish a universal ridge advantage.

## New paired accounting

Read `private_predictions.pt` on CPU; compare each condition's local argmax
with the same saved `local_y` (3,072 labels). No class remapping is necessary
for within-occurrence correctness. All condition accuracies reconcile with
`metrics.json`. Counts are query occurrences, not independent accounts.

| Other condition | Both correct | Ridge correct / other wrong | Ridge wrong / other correct | Both wrong |
|---|---:|---:|---:|---:|
| Native | 1305 | 529 | 153 | 1085 |
| Value replacement | 470 | 1364 | 316 | 922 |
| Native calibrated | 1030 | 804 | 468 | 770 |
| Value calibrated | 1186 | 648 | 621 | 617 |
| Ridge calibrated | 1180 | 654 | 445 | 793 |

The small 27-occurrence deficit of calibrated value replacement is the net of
**621 ridge errors corrected and 648 ridge-correct decisions lost**. Its
predictions disagree with ridge on 1,269/3,072 occurrences (41.31%), not on
just the 0.879 percentage-point accuracy gap. Native, meanwhile, corrects
153 ridge errors but loses 529 ridge-correct decisions.

These are paired decision comparisons, not causal descriptions of ridge
being processed by the metagraph: ridge and native are different readouts.
Error complementarity alone does not imply a usable selector or ensemble;
an oracle choosing between them would consume unavailable query labels.
There is no statistical significance claim from these occurrence counts.

## Decision

### Follow-up: support-fitted sign reversals explain the ranking loss

Joined all 128 saved calibration records to prediction episode IDs and computed
local binary margin AUC before/after calibration. For every method and episode,
positive slopes preserve AUC and negative slopes produce `1 - AUC`, with
maximum numerical discrepancy 1.12e-16. This is exact score accounting, not a
newly fitted model or revised calibration protocol.

| Readout | Negative-slope episodes | Query AUC before / after, negative subset | Ridge-only / calibrated-only correct, positive subset | Ridge-only / calibrated-only correct, negative subset |
|---|---:|---:|---:|---:|
| Native | 63/128 | .534392 / .465608 | 128 / 109 | 676 / 359 |
| Value replacement | 33/128 | .616723 / .383277 | 260 / 495 | 388 / 126 |
| Ridge | 49/128 | .631708 / .368292 | 64 / 76 | 590 / 369 |

For value replacement, the 95 positive-slope episodes average .623879 AUC
both before and after calibration. The 33 negative-slope episodes account
for the entire mean within-episode AUC decline, .622034 to .561849. Its net
accuracy deficit against ridge is the sum of +235 correct occurrences in
positive-slope episodes and -262 in negative-slope episodes, totaling -27.
This is a subgroup decomposition, not evidence that selecting a fallback by
slope would outperform ridge on untouched data. Subgroups and outcomes have
now both been inspected; no such prospective test is claimed or nominated.

The interpretation of the 41% disagreement is therefore more specific than
generic complementary evidence: a substantial part comes from a support-fit
that reverses useful query ordering. The precise cause of the wrong signs
remains open: six-example estimation noise, differences between held-out
support and recipient query score distributions, and changes in context size
are competing explanations. Do not call all of these a distribution shift
established by this analysis. The result also affects calibrated ridge, so it
is not unique to the value intervention or proof of a metagraph-specific bug.

Calibration SHA256:
`cb859620757c862485d0ca205da98996f03eee265ef77947aabc66f32b11bce6`.

### Bounded fold-offset check: rejected explanation

Inspected the producing `support_boundary_episode.py`,
`support_boundary_folds.py`, and `support_boundary_calibration.py`. Each of
three folds holds exactly one example from each class, paired in their saved
within-class order. Verified all 128 private episode files against the
producing receipt before reading their saved out-of-fold support margins.

Let fold k's two margins be m(k,1), m(k,0). The difference between pooled
class-mean margins is exactly the mean of the three paired differences.
Adding any shared offset to both scores in a fold cancels from that quantity.
For the balanced, standardized, strictly convex regularized logistic fit,
the profiled objective's slope derivative at zero has the opposite sign of
this class-mean difference (the optimal intercept at zero is zero). Therefore
the optimum slope has the same sign as the mean paired difference. This is
a sign result only: fold offsets can still change scale, fitted magnitude,
intercept, and subsequent predictions.

All 384 saved fits obey this sign identity. Among negative-slope episodes,
the counts with one/two/three negatively ordered held-out pairs are:

| Readout | One negative pair | Two | Three |
|---|---:|---:|---:|
| Native | 6 | 30 | 27 |
| Value replacement | 11 | 19 | 3 |
| Ridge | 3 | 28 | 18 |

Thus a between-fold shared-offset artifact cannot explain the negative slope.
In 11 of the 33 value episodes, one negatively ordered pair outweighs two
positively ordered pairs; in the other 22, most or all pairs are negatively
ordered. This is evidence about the actual tiny calibration sample, not a
proof of sampling noise as the generative cause. Pairwise centering cannot
repair the sign problem. Do not launch that experiment.

Receipt SHA256:
`274991efabb44dfba05e96af4353c50e3324f6d7bc6fe98ba6dbb71f6debb11f`.
This closes the bounded calibration audit. It explains the failed repair,
not the source-dependent graph-context effect; further calibration variants
would move away from the paper's central unresolved question.

Do not repeat support calibration as an untested explanation of deployment
utility. Do not equate similar aggregate accuracy with similar decisions.
Retain the mechanism claim and the failed practical repair separately. Any
later source/schedule analysis should retain corrections and corruptions
alongside mean readout gaps; otherwise substantial error exchange can be
hidden. No new experiment or selector is nominated by this audit.

Saved prediction SHA256:
`5ea31d13b0d21f14b9d0d808350a53b57672bbebbf852e2e50ad5477df994371`.
Producing revision: `1b49e2f9ea23ac68619714f42cfa86960b2d108a`.
Analysis worktree `.worktrees/role-topology`, branch
`codex/role-topology-interactions`, HEAD `30df8cb5`. Raw tensors remain private
on Tucker; this note remains uncommitted.
