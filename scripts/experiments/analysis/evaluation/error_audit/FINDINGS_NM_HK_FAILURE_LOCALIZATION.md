# HK NM failures: membership ambiguity and class-reference construction

8 September 2026. Read-only follow-up to the canonical source-stage comparison.
No training, new graph sampling, encoding or model forward passes. Two questions
are kept separate: what the original benchmark calls an error, and why changing
supports fails even when the same encoded inputs admit a correct decision.

## What has become clearer

1. The original HK benchmark has real overlapping memberships, but most errors
   are not alternative valid test neighbors. Of 50,464 native HK errors, 3,315
   (6.57%) select another `static_test` neighbor; another 13,403 (26.56%) select
   a training/validation neighbor. The remaining 33,746 (66.87%) select a
   candidate with no recorded edge in any of the three views.
2. Under nearest-support replacement, 36 of the 71 previously persistent
   failures are solvable by the common cosine head, while the native head solves
   only five. There are **31 cases with the same encoded inputs where cosine
   succeeds and the native head fails**. These are selected interventions, not
   31 newly explained original benchmark errors.
3. A controlled saved-score intervention localizes the nearest-support outcome
   changes to the true-class score: replacing only that entry reproduces the
   native correctness outcome on **all 200 cases**. Moving only rival scores
   rescues none of the 100 original failures and breaks two correct controls.
   Competitor movement is not needed for this set's rescues or breakages.

Together these results narrow a mechanism to investigate: how the learned
metagraph converts support representations into a class reference. They do not
identify the training cause, prove a universal readout defect, or demonstrate a
benchmark improvement.

## 1. Full canonical population: what does an error predict?

Use unchanged native predictions for all 512 HK test episodes: 61,440 query
occurrences, 6,233 observed query nodes, 30 classes, three supports and four
queries per class. Check each predicted anchor–query pair against the complete
cached canonical edge views, with the sampler's undirected membership convention.

| Native HK error predicts | Errors | Fraction of 50,464 errors |
|---|---:|---:|
| Another test-edge neighbor | 3,315 | 6.57% |
| A training-edge neighbor | 10,962 | 21.72% |
| A validation-edge neighbor | 2,441 | 4.84% |
| No recorded edge in any view | 33,746 | 66.87% |

Thus 33.13% of assigned-anchor errors still select a recorded graph neighbor.
That is exact accounting of the stored graph, **not** a fraction whose training
cause has been established. The benchmark asks for the sampled generating
anchor, so these predictions remain wrong under its original definition.
Training/validation neighbors are not held-out test positives; their counts
must remain separate from alternative test positives. Absence of a recorded
edge is not a claim about whether two real users have any relationship.

There are multiple valid test anchors among the 30 candidates in 28,407/61,440
occurrences (46.24%). This is larger than the earlier 23.12% query-role collision
rate because complete adjacency also includes positive pairs not drawn into
query roles in that episode. Mean valid candidate count is 1.916 for test edges
and 6.653 for all three views. Accordingly, accepting any graph neighbor changes
both the task and its chance baseline: native's descriptive neighbor-hit rate
is 45.07%, versus 22.18% for a uniform candidate choice, rather than a 3.33%
single-label chance rate. **45.07% is not a revised benchmark accuracy.**

The foreign Ukraine checkpoint has a 42.73% any-graph-neighbor hit rate on the
same HK inputs, versus its 11.68% assigned-anchor accuracy. Native HK still wins;
this does not establish that its advantage comes from memorizing role bias or
that the sampling objective is invalid.

At the support interface, 23,754/46,080 support occurrences are also linked by
a test edge to a rival candidate anchor. The one-hot construction marks those
53,885 rival relations negative (of 1,336,320 negative support edges). This is
inconsistent with interpreting the signs as exhaustive graph adjacency, but
consistent with their actual meaning of membership in the sampled episode
class. Whether these overlapping memberships cause harmful training gradients
has not been tested.

6,658 of 8,843 all-four-wrong native classes have no alternative-test-positive
prediction among their four queries. None of the 8,843 has all four errors
accounted for by alternative test positives. Overlapping memberships therefore
do not simply relabel away the concentrated class failures.

The observed-pair-only audit gave a lower bound of 2,909 HK alternative-positive
errors. Complete adjacency raises it to 3,315. Ukraine's retained observed-pair
count, 528/34,317, is still a **lower bound**, not an exact cross-target comparison.

## 2. Selected support interventions: evidence can become available without recovery

Reuse the existing fixed candidate banks and every saved head prediction.
Nearest means the three largest query-to-support cosines within the capped bank;
query inputs, rival support inputs and each candidate's sampled context stay
fixed. Selection uses the known true class.

| Originally failed population | Cases | Native correct, nearest | Mean-cosine correct, nearest | Cosine correct but native wrong |
|---|---:|---:|---:|---:|
| All selected failures | 100 | 25 | 62 | 38 |
| Previously persistent failures | 71 | 5 | 36 | 31 |

On original supports, the mean-cosine head solves zero of the 100 selected
failures. The new comparison thus requires **both changed support evidence and
a different decision rule**. It does not show that those original encoded
inputs already contained a cosine-solvable answer. Prototype sensitivity gives
61/100 and 35/71 nearest-support successes, with 36 and 30 native-failed cases
respectively solvable by that head.

Of the 66 persistent cases still wrong under native nearest selection, 31 are
cosine-solvable and 35 remain wrong under both heads. Because nearest maximizes
this fixed mean-of-three-cosines score within the bank, a different triple from
that same bank cannot improve that score. This is a limit of a particular head,
bank and frozen contexts, not a proof of absent information or irreducibility.

Nearest also increases cosine in 88/100 correct controls, yet 42 of those 88
break under the native model. Larger learned cosine is not sufficient to preserve
the native decision. No deployable selection rule or full-benchmark gain follows.

## 3. Which scores move when the true supports change?

The checkpoint has `S,U,M`, one M layer, no final-back layer, and fixed BatchNorm
statistics at evaluation. Query-to-label messages are excluded; query nodes
receive initial label vectors and self messages. Thus replacing support rows
cannot change the query's final vector in this architecture. Supports can change
all class references because `meta_gnn_pos_only=False` includes signed negative
support-to-rival-label messages.

Let A be the original native 30-way score vector and B the vector after support
replacement. Construct two hybrids without new inference:

- True only: use B's true-class score and A's 29 competitor scores.
- Rivals only: use A's true-class score and B's 29 competitor scores.

These are exact interventions on saved scores. Under the verified single-layer
architecture they correspond to swapping the true versus rival final class
references with the fixed query vector. Hybrid references need not be realizable
by one natural support set. No counterfactual model forward or post-M tensor
capture was performed in this follow-up.

| Condition | Rescues / 100 original failures | Retains / 100 correct controls |
|---|---:|---:|
| Full nearest replacement | 25 | 46 |
| Only true-class score changed | 25 | 46 |
| Only competitor scores changed | 0 | 98 |

Full and true-only correctness agree **case by case**, not merely in aggregate.
None of these hybrids is within 1e-6 of a top-1 tie. For the 31 persistent
cosine-correct/native-wrong cases, true-only remains wrong on all 31. Restoring
the original rival references therefore does not recover this residual.

The native true-score change on originally correct nearest cases averages
−1.616 logits; the best-rival change averages +.011. On persistent nearest cases
the corresponding changes are −1.198 and +.015. Scores retain the checkpoint's
fixed scale; these are not calibrated likelihood comparisons. The identity
`margin change = true-score change − best-rival-score change` holds to 1e-12.

The earlier random-replacement cache gives a consistent but imperfect pattern:
true-only gives 62/500 rescues versus 61/500 full, and retains 168/500 controls
versus 163/500 full. Rival-only gives zero rescues and retains 493/500 controls.
The separate fixed-bank random condition gives 60/500 versus 62/500 rescues and
171/500 versus 170/500 retained controls. These are correlated draws and distinct
candidate protocols, not independent replications.

This isolates the affected decision pathway, not attention versus value
projection versus normalization within that pathway. A mean-cosine classifier
and the native reference transform use different geometries. The evidence does
not require their score changes to be monotone for every valid support set.

## Working mechanism and the next discriminating question

HK NM combines overlapping relational memberships with an exclusive sampled
class task. Its original representations often fail to separate the assigned
class; some controlled support changes make that class separable under a fixed
head, while the native class-reference construction still fails to use it.
This is a supported two-stage account, with a large unexplained original-error
residual. The original full-stream cosine head recovers only 6.21% of native
errors and loses more native successes than it gains, so removing the metagraph
is not supported.

The next causal question is whether the true-class failure under changed supports
comes from **attention selecting/weighting the supports, the vectors projected
from those supports, or downstream reference normalization**. Existing cached
pre-metagraph inputs permit a bounded key/value and reference decomposition.
Retain original successful cases and the foreign checkpoint/Ukraine controls so
this remains a test of source-trained behavior. Prior HK-to-classification K/V
results cannot be transferred to NM as if this test had already been run.

A targeted model change would need to improve this support-to-reference mapping
while preserving native successes. Extra support randomness alone is not a new
mechanism: the sampler already redraws anchor members and their graph contexts
during training. The historical `lowest_sorted` retention creates systematic
support/query role bias within those stochastic draws. Neither paired-support
consistency training nor an overlap-aware objective has been validated by this
follow-up. Exact exposure and gradient-conflict effects during HK pretraining
remain unknown.

Estimated next bounded metagraph-only investigation: setup 30–60 minutes;
compute 1–3 minutes on an available owned Tucker GPU, plus artifact I/O. This is
an estimate for a proposed investigation; **no such run was launched here**.

## Artifacts and checks

- [Complete HK membership accounting](data/canonical_split/nm_hk_full_memberships.json),
  [observed-pair lower-bound sensitivity](data/canonical_split/nm_known_memberships.json),
  [membership helper](audit_nm_known_memberships.py).
- [Score interventions and selected-case accounting](data/canonical_split/nm_hk_score_paths.json),
  [score helper](audit_nm_hk_score_paths.py).
- All input CSV hashes match the prior canonical receipts. Sample/anchor joins,
  model-predicted anchor IDs, all 42,023 observed ordered HK test pairs, and
  all four original model-target accuracy totals are checked. Complete cached
  train/validation/test views remain disjoint as undirected pairs; an independent
  packed-integer edge join reproduces all native HK error counts.
- Both large logit-cache SHA-256 values match their receipts. Metadata-only
  extraction reads just their logit storage blocks, with no graph or embedding
  materialization; the private compressed export is 576,527 bytes at
  `/private/tmp/nm_hk_score_cache.npz`. All 3,800 cached rows × three heads
  reproduce saved predictions, correctness, ranks and margins. No gates were
  relaxed to produce these results.
- Complete edge cache:
  `/dataMeR1/phil/gfm/error_audit/nm_hk_support_extremes_20260908/hk_canonical_views_private.pt`;
  logit sources are the existing mechanism and extremes private archives.
  No private node-level data or logits are added to git.
- Worktree `/Users/philipp/projects/gfm/prodigy`, branch `main`, initial HEAD
  `24b8b624`. Another worker advanced HEAD to `fc0b9959` during this review;
  no commit was made here. New helpers, aggregates and this report are
  uncommitted. Existing unrelated work was preserved. No cluster worktree changed.
