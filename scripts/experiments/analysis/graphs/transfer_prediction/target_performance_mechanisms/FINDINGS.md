# Target performance: what the completed mixture lattice establishes

Audited 2026-09-06 UTC. This is the first descriptive stage of a mechanism study.
No new model training or GPU evaluation was performed. The analysis uses 270
classification cells and 414 NM cells, with training seed 0 throughout.

## The explanatory question has two different scales

For classification, an orthogonal decomposition of the complete 54-model by
five-target AUC matrix assigns 95.34% of its observed squared variation to target
means, 1.69% to model means, and 2.96% to model-target interaction plus noise.
Among the nine specialists alone the split is 88.97%, 4.39%, and 6.64%.

These are descriptive in-sample sums of squares, not causal variance fractions.
The large target term mainly reflects the distance between easy and near-chance
tasks. It does not make the remaining differences unimportant: singleton
interaction RMSE is 0.0472 AUC, and individual target gaps can be much larger.

| Classification target | Mean singleton AUC | Mean pair AUC | Mean LOO AUC |
|---|---:|---:|---:|
| COVID-political | .8416 | .9017 | .9275 |
| Election2020-political | .9761 | .9854 | .9889 |
| Facebook Page Reference | .7112 | .7330 | .7729 |
| TwiBot-20 | .5880 | .6011 | .6241 |
| Ukraine-suspended | .4936 | .4969 | .4885 |

Every cell uses 128 episodes, but query totals differ: 3,072 for COVID-political
and TwiBot-20; 1,024 for Facebook; 256 for Election and Ukraine-suspended. Do not
interpret their fluctuations as equally precise. This grid lacks training-seed
replication and saved query-level predictions.

NM has a different decomposition. On its nine-by-nine specialist AUC matrix,
target means explain 34.98% of observed squared variation, model means 33.13%,
and interaction plus noise 31.89%. An explanation of NM transfer cannot simply
be promoted to an explanation of downstream classification.

## Size is a clue; the new crossed comparisons expose compatibility

Mean specialist classification AUC over the identical five-target panel ranks
Ukraine (.7678), TwiBot-20 (.7635), and COVID (.7563) first. TwiBot-20 has about
163k nodes, compared with Ukraine's 10.4M and COVID's 23.0M. Raw node count is
therefore insufficient to explain donor ordering. This is descriptive and
includes each labeled source's own target; the next comparison is wholly foreign.

For each target, compare `(Ukraine, partner)` against `(TwiBot-20, partner)`.
Keep the partner, target, source count, seed, and nominal update budget fixed;
exclude partners equal to either donor or the target. There are six partners:

| Target | Mean Ukraine-pair minus TwiBot-pair AUC | Range over six partners | Sign |
|---|---:|---:|---|
| COVID-political | +.01722 | +.00556 to +.04491 | Ukraine wins 6/6 |
| Facebook Page Reference | -.02958 | -.06136 to -.00544 | TwiBot wins 6/6 |

Thus a universal donor score leaves important target dependence unexplained.
These are replicated compositions within one seed, not six independent training
replications of the same contrast. They provide precise cases for diagnostic
replay and future seed replication.

Across nine donors, edge count correlates with mean classification AUC at
Spearman .833, largest weak-component fraction at .783, and node count at .583.
Density is -.200 and approximate clustering -.217. These correlated, full-graph
statistics do not identify a mechanism. In particular, "more interconnected"
needs an operational definition; density, component coverage, degrees, and the
distribution of sampled local neighborhoods are different quantities.

Production uses balanced source draws and 2,500 updates at batch size four.
Large graphs get no greater expected episode allocation. They may still supply
more distinct usable examples, different sampled-node counts, easier or more
informative pseudo-label relations, or gradients with greater effective influence.

## Similarity still provides only part of the answer

On foreign singleton classification cells, mean within-target Spearman rho is
-.548 for center-plus-neighbor projected Frechet and -.543 for neighbor-mean
Frechet. The former ranges from -.857 to -.095 across targets. Raw feature
proxy-A from the earlier artifact is -.427 here. These are classification results;
the stronger -.738 headline in similarity-vs-transfer v2 refers to NM AUC.

The metrics use previous graph samples. They do not measure the actual anchor,
positive-member, support, query, and context distributions consumed during the
new training runs.

## The best-constituent envelope is descriptive, not an interaction mechanism

Across 140 foreign classification pair cells, mean pair minus best specialist
AUC is -.01076; 28.6% are above that reference. Across 252 foreign NM pair cells,
the mean is +.00388 and 51.6% are above it. LOO models fall below their best
remaining specialist in all five foreign classification cells and all nine NM
cells, by mean .00932 and .00964 AUC respectively.

The constituents received all 10,000 episodes when trained alone, versus an
expected 5,000 each in pairs and 1,250 each in LOO. These comparisons combine
source composition, exposure, optimization, and checkpoint noise. They do not
identify interference or synergy. There is no empty-data or matching all-nine
classification reference in this imported panel, and the NM LOO table contains
only the omitted target for each model. Do not infer full Shapley values or
same-target source-deletion effects from those missing comparisons.

## A reproducible sampler property needs investigation

The production `_sample_center_members` takes 70 one-hop walk endpoints for
seven requested members, applies `torch.unique`, and keeps the first seven.
Those unique endpoints are sorted by node ID. The collator assigns the first
three members to support and the remaining four to queries.

The included reproduction calls the actual method on a fixed walk with ten
unique endpoints. It selects nodes `[0,1,2,3,4,5,6]`. Relabeling the same endpoints
consistently and mapping them back produces `[9,8,7,6,5,4,3]`. Selection therefore
depends on numerical node ordering, and support/query roles also depend on that
ordering. The method was checked in the producing Tucker pair worktree as well
as this checkout.

This proves a sampling property, not its magnitude or causal relevance on the
real graphs. It motivates an index-permutation control and real draw audit before
interpreting time span, eligible degree, or neighborhood diversity. There is also
no cross-class member exclusion in the standard confined-source path: collision
and conflicting pseudo-label rates should be measured, not assumed absent.
Production code was left intact so historical streams remain reproducible.

## Existing evidence that the next study should reuse

- [Identity-disjoint evaluation](../../../transfer/matrices/prodigy_nm/identity_disjoint/entity_disjoint_eval/FINDINGS.md):
  removing overlapping nodes from target centers and all context lowered mean
  NM accuracy from .3192 to .2861 across the tested 18 cells, with a .1057 drop
  for Ukraine to COVID. Transfer remained well above chance. Target distributions
  changed, so this does not isolate memorization from graph damage or difficulty.
- [Identity audit](../../overlap/identity_overlap_audit/FINDINGS.md): COVID covers
  2020-01-23 to 2021-02-21, Ukraine 2022-02-22 to 2022-04-28, and Midterm
  2022-10-01 to 2022-10-23. Strong Ukraine transfer alongside its much shorter
  collection window makes duration alone an inadequate donor ordering rule.
- [Path/feature diagnostics](../../structure_features/path_feature_coupling/FINDINGS.md):
  full-data label probes differ sharply by target; they are useful signal
  references but are not matched few-shot baselines.
- [Adaptation-efficiency study](../../../evaluation/adaptation_efficiency/FINDINGS.md):
  raw-feature and learned-feature baselines already exist in another protocol.
  Reuse their implementations, then rerun on the exact diagnostic episodes.
- The `nm-interventions-overnight` worktree already contains a broad source-held-out
  NM intervention campaign. Its checkpoint-selection rule, training implementation,
  and NM-only outcome differ from this lattice. Audit and reuse relevant arms;
  their current rankings do not establish a downstream classification mechanism.

## What is available for the next measurement

The nine specialist terminal checkpoints were read on Tucker's CPU: their tensors
are finite and each retains two sets of BatchNorm running variances. This confirms
that source-dependent normalization state is a measurable candidate, not evidence
that it causes the donor gap. See `data/checkpoint_buffer_audit.json`. Comparing
buffer magnitudes across differently learned coordinate systems does not identify
target compatibility; the calibration intervention is the relevant test.

Checkpoints at 100/300/900/2500 and exact model paths are registered. A sampled
classification result directory retained aggregate metrics and score summaries;
the production launcher did not enable prediction export. Source-gradient logging
exists from a separate short experiment, but it compares source NM gradients to
one another rather than explaining target decisions. The historical multiworker
training stream was not archived as realized episodes; resampling the same
distribution must not be called recovery of the exact consumed examples.

The research protocol is in the sibling paper workspace:
`/Users/philipp/projects/gfm/paper/planning/target_performance_research_design_2026-09-06.md`.
Its first stage distinguishes target signal, learned representation, support use,
and sampling effects before committing to a coverage, scale, or optimization account.
