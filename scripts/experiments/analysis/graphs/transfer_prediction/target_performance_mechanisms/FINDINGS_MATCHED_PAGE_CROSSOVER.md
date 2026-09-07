# Matched page trajectory: catch-up, not absolute deterioration

7 September 2026. Advisor/researcher synthesis from completed saved results.
No new model forwards. Private; main manuscript unchanged.

## Absolute performance changes the causal question

Hong Kong source, same cached page target episodes across five saved steps.
Means over three training seeds and two streams (streams are not seeds).
Within-episode AUC from native logits, not final float64 reconstructions:

| Step | Intact AUC | Suppressed AUC | Intact NLL | Suppressed NLL |
|---|---:|---:|---:|---:|
| 0 | .479574 | .479614 | .693197 | .693197 |
| 100 | .538330 | .504232 | .739263 | .693198 |
| 300 | .640544 | .632080 | .675602 | .673481 |
| 900 | .659993 | .662598 | .831923 | .738117 |
| 2500 | .691732 | .700602 | .836483 | .709814 |

Both paths improve in ranking. Suppressed inference catches up and overtakes
intact inference; the sign reversal does not show absolute deterioration of
the intact classifier. The pooled reversal holds in 6/6 matched comparisons;
the within-episode step-100 effect is negative in 5/6, and the step-2500 effect
positive in 6/6. Do not substitute the pooled unanimity for within-AUC evidence.

## Geometry is consistent with initially ineffective suppressed references

Mean distance between the two normalized final references, 256 episodes per
seed (both streams), alongside intact-to-suppressed contrast orientation cosine:

| Step | Seed | Intact separation | Suppressed separation | Orientation cosine |
|---|---|---:|---:|---:|
| 100 | 0 | .0577474 | .0045466 | .0473463 |
| 100 | 1 | .0588322 | .0069169 | .2472920 |
| 100 | 2 | .0836967 | .0065843 | .1513301 |
| 2500 | 0 | .5117186 | .3835052 | .9154412 |
| 2500 | 1 | .5547712 | .4046461 | .9106422 |
| 2500 | 2 | .5556590 | .3791403 | .9319900 |

Near-coincident early suppressed references coincide with near-chance ranking.
At the later step both conditions distinguish classes and have similar contrast
directions. Separation alone is not an explanation of AUC: positive rescaling
cannot change episode ranking. These observations motivate locating the
learned crossover, not asserting that increasing separation causes it.

## One nominated hypothesis discrimination, before new prototype outcomes

Independent advisor/reviewer recommends testing whether the learned reference
constructor comes to misuse support context that remains useful to a fixed
prototype. On exactly the three matched seeds, two streams, steps 100 and
2500, compare the already-defined fixed normalized-support prototype with
native inference. Fixed intact U1 queries within each checkpoint; intact versus
suppressed support U1 vectors. Average individually unit supports by class,
normalize class means, cosine-score queries. No fitting, scale choice, or
selection. Preserve all six comparisons rather than select an agreeing seed.

Hypothesis: prototype context utility remains positive at both steps while
native utility changes sign. This would locate a learned incompatibility
downstream of the tested prototype geometry. If prototype utility instead
reverses with native utility, reject that constructor-specific explanation:
the crossover is already visible with a fixed readout of encoder outputs.
Mixed prototype results are inconclusive, not grounds for checkpoint search.
Neither outcome alone constitutes a complete mechanism or a deployment method.

The previous early political prototype improvement makes a constructor-only
account uncertain; do not assume it before running the page comparison.

Read-only Tucker inventory found aggregate geometry and predictions, but no
saved U1 activation directory in the scale, final-contrast, or dose runs.
The nominated comparison therefore needs a bounded replay of existing inputs
and weights, not merely recomputation from currently listed aggregates. It
has not yet run. Do not claim an existing artifact contains the needed vectors.

Sources: `data/classref_decision_20260907/training_ranking_cells.csv` and
`data/classref_contrast_20260907/geometry.json`. Local worktree
`.worktrees/role-topology`, branch codex/role-topology-interactions, 48907a3b.

## Completed prediction test: the nominated account is contradicted

Runner 8ccd3686, isolated Tucker `prodigy-page-prototype`, output
`log/page_prototype_20260907`. All 12 paired cells / 48 metric cells completed.
Each cell contains 128 page episodes, 1024 queries (not the political panel's
3072 queries). Both native endpoints are bit-exact with original saved logits;
U1 and final queries are bit-exact across context conditions. All cached input
hashes and checkpoint weight digests match. The existing episode_probe function
implements the fixed prototype, here evaluated in float64. No fitting.

| Step / readout | Intact within AUC | Suppressed within AUC | Intact accuracy | Suppressed accuracy |
|---|---:|---:|---:|---:|
| 100 / native | .538330 | .504232 | .504720 | .501953 |
| 100 / fixed prototype | .570068 | .649333 | .540365 | .630371 |
| 2500 / native | .691732 | .700602 | .637858 | .649740 |
| 2500 / fixed prototype | .764323 | .781901 | .695475 | .707682 |

Means over the six seed/stream cells, not independent query estimates.
Prototype suppression improves within-AUC AND accuracy in all six comparisons
at both steps. Within-AUC deltas, in points, retain every cell:

| Stream | Seed | Step100 native | Step100 prototype | Step2500 native | Step2500 prototype |
|---|---|---:|---:|---:|---:|
| Original | 0 | -5.713 | +8.057 | +.342 | +.586 |
| Original | 1 | -3.149 | +3.516 | +.586 | +1.416 |
| Original | 2 | -5.151 | +2.148 | +1.514 | +2.881 |
| Fresh | 0 | -3.442 | +13.330 | +2.100 | +2.002 |
| Fresh | 1 | +.195 | +9.521 | +.732 | +1.709 |
| Fresh | 2 | -3.198 | +10.986 | +.049 | +1.953 |

The prespecified hypothesis was that prototype context utility stays positive
(suppression hurts) while native utility reverses. It fails: prototype context
utility is negative (suppression helps) at both steps, in every comparison.
This is neither the nominated explanation nor the alternative of a prototype
sign reversal alongside native inference. Preserve this third outcome.

## What training changed: more precise interpretation

At step100 the suppressed encoder outputs already support .649 prototype AUC
and .630 accuracy, while native inference produces .504/.502. Thus near-chance
native suppressed behavior cannot be explained by an absence of usable class
information in those support/query embeddings. At step2500 both readouts
benefit from suppression, though the fixed prototype remains stronger.

The native crossover is compatible with improved use of an already useful
suppressed representation, rather than a constructor progressively misusing
context that remains beneficial to the prototype. This comparison does not
isolate the class-reference update alone: replacing native inference by the
prototype also changes query-space processing and other readout operations.
Call it an encoder-versus-native-readout distinction, not a proven defect in
one value projection. Do not infer that a particular parameter changes causes
the trajectory without a targeted intervention.

This is explanatory progress against an important competing account, not a
complete mechanism or an 8+ contribution established by one more contrast.
Do not add checkpoint search or prototype variants after this result.

Artifact SHA256s:
- protocol: c50ade4e9822f32fbdf210fc93a4a0af4d95cc731098a191206d5e57f9b9241b
- metrics: 68277036e12514742fb31dd1ef1b329c2d8f71dc485bf5a985a08934139f6586
- receipts: fbbfcb1293344af16de2dd3ecf4baaec109c98aceb760e1ff585c079049e5a28
- DONE: 946145b50f70dc4830173e12e2b55a8eae967d6e7fcfdb8ce2ed46d1cf8f1646

## Episode-level audit of the available-information gap

Saved suppressed predictions only; zero new forwards. Across 768
seed-by-episode evaluations at each checkpoint (the same episode streams
repeat across seeds, so these are not 768 independent examples):

| Comparison | Step100 | Step2500 |
|---|---:|---:|
| Prototype AUC exceeds native | 492 | 443 |
| Prototype AUC below native | 218 | 133 |
| Equal | 58 | 192 |
| Native below .5 AUC | 349 | 124 |
| Prototype above .5 AUC | 536 | 683 |
| Prototype perfect, native at most .5 | 37 | 3 |

These counts show the mean gap is not one exceptional episode, but also that
the prototype does not dominate every episode. The final row is a descriptive
cutoff selected after the aggregate result, not a new confirmatory test.

Concrete first-listed disagreements from seed0/fresh at step100:

- Episode1, class indices [24,7] (MEDIA / POLITICAL_ORGANIZATION): prototype
  AUC1.0, native .4375; prototype5/8 and native4/8 correct. Perfect ranking is
  not perfect threshold classification.
- Episode18, [14,20] (LOCAL / CHARITY_ORGANIZATION): prototype AUC1.0 and8/8
  correct, native .375 and4/8 correct.
- Episode24, [6,23] (POLITICAL_PARTY / ENTERTAINMENT_SITE): prototype AUC1.0
  and8/8 correct, native AUC0.0 and4/8 correct. Native completely reverses
  ordering while retaining chance threshold accuracy.

These are post-hoc illustrative cases, not prevalence estimates or independent
semantic label validation. Names are resolved from the graph's recorded
`label_names` in `/dataMeR1/phil/data/facebook_page_reference/graphs/`
`page_reference_graph.meta.json`. This audit does not inspect the raw bios or
establish which profile feature caused a decision.

## Coordination boundary

The separate live `isolation-full` job belongs to Tucker worktree
`prodigy-encoder-solver-isolation`, local branch/worktree
`codex/encoder-solver-isolation` at `/private/tmp/prodigy-encoder-solver-isolation`.
Its recorded design compares native, joint native+ridge, gradient-isolated,
and ridge-only training under matched blocked/interleaved schedules. That
already tests a relevant encoder/solver co-adaptation hypothesis. Its findings
file at ec1fbfb3 reports only implementation/parity checks, not target outcomes.
Do not duplicate that training study or treat smoke results as confirmation
of the present interpretation. No change was made to its checkout or job.

## First-batch check of simple support-label neglect

Four read-only CPU forwards with existing kv_forward: seed0, original batch0,
steps100/2500, intact/suppressed. Actual saved attention is summed over incoming
matching-label, opposite-label and self edges, then averaged over eight class
nodes and eight heads. There are four episodes in this batch.

| Step / condition | Matching-label mass | Opposite-label mass | Self mass |
|---|---:|---:|---:|
| 100 intact | .639037 | .303787 | .057176 |
| 100 suppressed | .639961 | .303052 | .056987 |
| 2500 intact | .691776 | .294759 | .013465 |
| 2500 suppressed | .692042 | .296445 | .011513 |

The implementation conveys support labels through the attention edge channel,
not through the value projection. These actual weights show substantial role
differentiation already at step100 and almost no mean role-mass change with
suppression. They argue against a simple first-batch story of uniform routing
or absent matching-label preference. They do NOT causally isolate use of the
label code from correlated content/receiver differences, rule out per-head
effects, or explain transported value directions. Do not claim that all128
episodes or all seeds have been checked by this small diagnostic.

The likely bottleneck is more specific than an attention budget that fails to
favor matching labels; weighted values can still construct ineffective
references despite that preference. This is a narrowed hypothesis space,
not a new predictive mechanism or a reason for an attention-component sweep.
