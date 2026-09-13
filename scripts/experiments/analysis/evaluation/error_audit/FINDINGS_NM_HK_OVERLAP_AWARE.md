# HK→HK NM: multi-positive evaluation and contradicted-negative masking

8 September 2026. Full 61,440-query canonical HK stream, unchanged native HK
checkpoint and cached pre-metagraph embeddings. No training, graph encoding,
sampling or GPU use. The protocol was frozen before inspecting outcomes.

## Candidate-set link-prediction evaluation

If NM is interpreted as link prediction, every candidate anchor joined to the
query by a held-out test edge is a valid answer. Under that definition, the
unchanged predictions score:

| Metric | Result |
|---|---:|
| Original assigned-anchor accuracy | 10,976 / 61,440 = 17.86% |
| Multi-positive top-1 accuracy | 14,291 / 61,440 = **23.26%** |
| MRR over all held-out-positive candidates | 41.39% |
| Hits@3 | 48.43% |
| Hits@5 | 63.81% |

The 3,315 added top-1 successes are exactly assigned-anchor errors where the
chosen candidate is another held-out test neighbor. This is a metric correction
for the LP interpretation, not a model improvement.

There are 33,033 uniquely answerable occurrences and 28,407 with multiple valid
test candidates. Accuracy on the unique subset is 21.84%. Of the ambiguous
rows, 13,594 have two valid candidates and 14,813 have at least three. A uniform
candidate has a 6.39% expected held-out-positive rate because the mean number of
valid candidates is 1.916, rather than the 3.33% single-label chance rate.

## Inference-only masking does not repair the model

For each negative support→label message, remove it when the support center and
candidate anchor are joined by a known training-view edge. This masks 225,875
of 1,336,320 negative messages across all 512 episodes while leaving embeddings,
positive/query messages, candidate sets, weights and decoder unchanged.

| Evaluation | Native | Training-edge mask | Change |
|---|---:|---:|---:|
| Assigned-anchor accuracy | 17.8646% | 17.8206% | −0.0440 pp |
| Multi-positive accuracy | 23.2601% | 23.2113% | −0.0488 pp |
| MRR | 41.3890% | 41.2845% | −0.1045 pp |

The mask recovers 508 assigned-anchor failures but breaks 535 successes. Under
multi-positive scoring it recovers 718 failures and breaks 748 successes. On
the uniquely answerable subset it recovers 259 and breaks 277. It produces a
small positive multi-positive balance only among queries with at least three
valid candidates (301 recovered, 280 lost), which is an outcome stratum rather
than a deployable selection rule.

The nondeployable all-view mask, which also uses validation/test adjacency,
performs worse: 17.70% assigned and 23.02% multi-positive accuracy, with
859 multi-positive recoveries versus 1,004 losses. More complete masking is not
a monotone remedy.

Within the 43,960-error residual, training-edge masking recovers 389 assigned
anchors. Almost all are existing near misses: 384/389 have native true rank 2–3;
none of the 13,970 rank-11–30 residual errors reaches its assigned anchor.
Always-wrong queries gain only 30 assigned recoveries. The intervention changes
decisions but does not address the persistent semantic-alias population.

## Interpretation

The evaluation problem is real if the claimed task is link prediction: nearly
half the occurrences contain multiple held-out-positive candidates, and the
single-label metric rejects 3,315 graph-correct top-1 predictions.

The forward-pass intervention is a controlled negative result. Merely deleting
currently contradicted negative messages is insufficient and slightly harmful.
This does **not** test whether contradictory labels damaged representation
learning: the checkpoint was already trained under the exclusive objective, and
the inference mask cannot undo those gradients. It also removes signals from a
readout coadapted to receive a fixed 3-positive/87-negative support pattern.

The next defensible experiment therefore requires training. Compare the original
objective with a multi-positive, overlap-aware objective that never labels a
known training neighbor negative, keeping source graph, sampled episodes,
checkpoint budget and evaluation episodes fixed. Primary evaluation should be
multi-positive; assigned-anchor and uniquely answerable results remain diagnostic.
This directly tests the gradient-conflict hypothesis that inference masking
cannot answer.

[Aggregate result](data/canonical_split/nm_hk_overlap_aware.json) ·
[subgroup accounting](data/canonical_split/nm_hk_overlap_aware_subgroups.json) ·
[runtime protocol](../../../setup/nm_hk_goal/README.md). Private prediction rows
are under `/dataMeR1/phil/gfm/error_audit/nm_hk_overlap_aware_20260908/`.
Runtime revision `63211b2d`, branch `codex/nm-hk-goal-20260908`, Tucker worktree
`/dataMeR1/phil/gfm/prodigy-nm-hk-goal-20260908`. Successful compute took 46.52
seconds on two low-priority CPU threads; no GPU was allocated.
