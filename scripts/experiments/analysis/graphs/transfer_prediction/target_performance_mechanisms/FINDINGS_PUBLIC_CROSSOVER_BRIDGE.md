# Public temporal crossover: compatibility, not a deteriorating inference module

7 September 2026. Read-only cross-worktree evidence reconciliation. Private;
no new experiment or manuscript expansion. The preceding turn made progress
by closing the redundant predictor branch and correcting manuscript source.

## Current external evidence

The public paired-state worktree has completed a temporal encoder/inference
crossover after the earlier role-topology manuscript was drafted. This is
distinct from its support/query edge-removal experiment. Read the full current
interpretation in the producing worktree's `public_prodigy_kg/FINDINGS.md`.

Independently read Tucker's `publickg_crossover_20260907/execution_status.json`
(complete), and recomputed all four means from `summary.json` episode rows.
Each condition has exactly 128 distinct episode ordinals; there are 512 rows.
The summary SHA256 at inspection was
`507e1169eb782be76223643ec3b81eb2ba8ca5a3faccf3d7cfb3805519f5d672`.
This audit validates row counts and aggregate arithmetic, not a new replay of
the model or independent rehashing of every saved prediction tensor.

| Encoder checkpoint | Inference checkpoint | Accuracy | Macro-F1 | OVR AUC | NLL |
|---|---|---:|---:|---:|---:|
| 2000 | 2000 | .774023 | .760251 | .982320 | .715275 |
| 2000 | 8000 | .780957 | .768567 | .982486 | .703857 |
| 8000 | 2000 | .731055 | .714448 | .973400 | .898459 |
| 8000 | 8000 | .750391 | .735312 | .976462 | .846310 |

Checkpoint filenames correspond to 2001 and 8001 completed optimizer updates.
The producing experiment holds each encoder's captured pre-metagraph data
embeddings fixed across inference conditions, owns normalization buffers with
their modules, and checks both diagonals against the earlier trajectory.
Public numerical parity is atol 1e-4, rtol 0, not social-runtime bit-exact replay.
The recipe retains native batch-statistics evaluation and the S2,UX,M2 model.

## What the contrast decides

Later inference improves accuracy under both encoders (+.6934 and +1.9336
points). Later encoder outputs reduce accuracy under both inference modules
(-4.2969 and -3.0566 points). Macro-F1, AUC and NLL agree in direction.
Thus the nominated early-inference-rescue prediction is contradicted.

The producing trajectory also reports fixed support-centered pre-metagraph
ridge accuracy .818945 early and .818066 late. Its nearly flat aggregate does
not mean unchanged representations: the paired transition audit records
1,313 correctness changes. The hypothesis is readout-dependent utility, not
information invariance. The crossover does not isolate support processing
from query processing or exclude normalization effects.

## Consequence for this paper

Do not describe the social fixed-checkpoint pathway localization as evidence
that inference weights progressively deteriorate during pretraining. The public
crossover supplies a concrete counterexample: inference changes compensate for
adverse encoder changes as evaluated by the native rule. A support-fitted
classifier can retain utility without either frozen learned inference module
retaining it.

The more promising common question is whether graph pretraining preserves
task-discriminative information while changing its compatibility with learned
class-reference construction. This is not yet a common proven mechanism:
the public crossover changes whole encoder/inference modules, while the social
K/V intervention fixes a checkpoint and changes only support context. Keep
these distinct causal scopes visible. Neither generic probe superiority nor
the name 'compatibility' is itself a novel explanation.

Advisor priority: assess the already-running normalization comparison and
fixed second-seed replication before deciding whether this public result should
anchor a revised argument. Do not duplicate those tests, search checkpoints,
restart the social geometry selector, or expand the manuscript around a
single-seed pattern. The goal remains unachieved, not narrowed to a diagnostic
paper by this decision.

## Live-job checks, not completion claims

At inspection, Tucker seed-one trainer PID 217056 was live under tmux
`publickg-seed1`, using owned GPU 2. Its output was
`prodigy-publickg-seed1/log/publickg_train_seed1_20260907`; no final target
result was claimed. The normalization comparison PID 260468 was also live,
using GPU 3, output `prodigy-publickg-paired-state/log/publickg_normalization_20260907`.
It had partial episode artifacts and no completed summary. Do not infer failure
from the missing summary while that process is live, or restart either run.

Producing local worktree `/private/tmp/prodigy-publickg-paired-state`, branch
`codex/publickg-paired-state`; crossover code `99524810`, current inspected
HEAD `98ba23d1`. Analysis note worktree `.worktrees/role-topology`, branch
`codex/role-topology-interactions`, HEAD `65946f82`. No source transport, remote
mutation, commit, or push was performed by this audit.

## Completion update: normalization does not eliminate the divergence

Rechecking the same normalization process found it terminal, with
`execution_status.json` reporting complete and a new full summary. Do not use
the preceding live snapshot as its current status. Summary SHA256:
`7eb3cee78da8f38c47975a29108d3d0f3e68d1a6a6c295568e4401344285b552`.
Independently recomputed all 32 aggregate metric means from 256 unique
step/episode rows (128 at each checkpoint), agreeing within 1e-12. There are
256 recorded artifact receipts; this audit did not independently rehash all files.

| Normalization / readout | Early accuracy | Late accuracy | Change, points |
|---|---:|---:|---:|
| Native batch statistics / native inference | .774023 | .750391 | -2.3633 |
| Native batch statistics / centered ridge | .818945 | .818066 | -.0879 |
| Checkpoint running statistics / native inference | .756543 | .701953 | -5.4590 |
| Checkpoint running statistics / centered ridge | .797461 | .808301 | +1.0840 |

Under running statistics, native macro-F1 falls 5.9927 points and OVR AUC
falls .7743 points; centered ridge rises 1.0073 and .0745 points respectively.
The original fixed ridge scale still gives substantially worse NLL than native
inference. Do not describe this as all-metric readout superiority.

The test changes only BatchNorm training flags, using each checkpoint's own
trained buffers with no recalibration. Unlike the crossover, encoder outputs
are allowed to change under the normalization intervention. The divergence
therefore is not explained away by using the released batch-statistics path:
it persists, and the accuracy divergence is larger, under frozen running
statistics. This does not show normalization is irrelevant, isolate which
buffers contribute, or establish that the temporal crossover signs also hold
under frozen statistics. That crossover was not run here.

Advisor implication: retain the public readout-dependent deterioration as the
leading complementary evidence for the compatibility account; the simple
evaluation-mode objection is insufficient. Await the already-designated seed
replication before elevating it to a general temporal claim. No new experiment
is launched by this update, and the social support-value effect remains a
distinct intervention rather than an asserted replication of this mechanism.

## Same-query compensation audit

Read-only saved-prediction analysis, no new forwards or model changes. Rehashed
all 512 crossover tensors and the 256 early/late trajectory tensors against
their producing receipts. Verified common source hashes, exact query labels,
native diagonal logit parity at atol 1e-4 and identical diagonal argmax decisions.
The 16 four-condition correctness patterns sum to 10,240 query occurrences and
reconstruct all four aggregate crossover accuracies exactly.

With early inference fixed, replacing early encoder outputs by late outputs
corrupts 1,243 previously correct occurrences. Replacing early inference by late
inference on those late outputs repairs 350 of these, leaving 893 wrong.
Across all late-encoder queries, later inference corrects 563 and corrupts 365,
giving the recorded net 198 additional correct decisions (+1.9336 points).
Thus compensation includes repairs on the very queries damaged by encoder
replacement; it is not solely a gain on an unrelated query population.

For 746 occurrences, both inference modules are correct with the early encoder
and wrong with the late encoder. In 371 of these, centered ridge is correct at
both checkpoints. This subset demonstrates a same-query readout discrepancy,
not just cancellation in aggregate ridge accuracy. It does not establish
information invariance, isolate the changed feature, or make a post-hoc subset
an independent confirmation sample. Other ridge decisions still change.

First three lexicographic examples in this latter subset:

| Episode / query | True local class | Early encoder: early / late inference | Late encoder: early / late inference | Ridge early / late |
|---|---:|---|---|---|
| 0 / 28 | 7 | 7 / 7 | 17 / 4 | 7 / 7 |
| 0 / 52 | 13 | 13 / 13 | 16 / 16 | 13 / 13 |
| 1 / 15 | 3 | 3 / 3 | 12 / 12 | 3 / 3 |

Local class indices are not semantic relation identities. These are prediction
traces selected by an explicit disagreement pattern, not semantic inspection or
representative examples of graph content. Repeated entities and shared supports
prevent treating 10,240 occurrences as independent experimental units.

Private aggregate record: `data/public_crossover_examples_20260907.json`.
Reproducible read-only calculation: `analyze_public_crossover_examples.py`, with
fixed Tucker-only source paths. It records all correctness patterns, not only
the favorable subset. No code or results were pushed by this audit.

## Actual relation examples behind the first three disagreements

Resolved the same three cases, without changing their selection. Saved query
outputs use mask-sorted data-point order, NOT shuffled prompt-sequence order.
Verified `y_true_matrix[query_indices]` exactly equals the saved output labels.
Mapped all 20 label vectors per episode to a unique, bit-identical relation
embedding in the released text-feature cache. Recovered entity pairs from
`center_node_idx`, `entity2id.json`, and the released alias dictionary. Aliases
are sometimes noisy; they are not independently verified canonical names.

- **Episode 0, query 52:** dataset aliases *The Rum Diary (film)* → *Colleen
  atwood*. The labeled relation is `film/costume_design_by`. Both inference
  modules predict it with the early encoder; both instead predict
  `film/production_companies` with the late encoder. Ridge stays correct.
  Costume-design supports pair film aliases with Ann Roth, Edith Head and
  Jean Louis. Competing production-company supports have tails aliased as
  DreamWorks Animation, Warmer Bros., and Universal Globe. These support
  examples make the confusion intelligible; they do not prove which feature
  or graph message caused it.
- **Episode 0, query 28:** *Searching For Debra Winger* → *Purangsu*, labeled
  with the film-cut release-region relation. Both early-encoder predictions
  are correct. With the late encoder, early inference predicts film genre;
  late inference predicts Netflix-genre-to-titles. Ridge remains correct.
  The release-region supports include tails aliased as Kenadian, Republic of
  Korea and Philippine archipelago. Preserve the supplied aliases rather
  than silently fixing them. Do not infer the query tail's identity from
  spelling alone.
- **Episode 1, query 15:** *President of the Assembly* → *Republic of China
  (1912-49)*, labeled with officeholder jurisdiction. Both early-encoder
  predictions are correct; both late-encoder predictions are business
  industry. Ridge stays correct. The correct-class supports include Prime
  Minister and Cyning head aliases; competing supports include Avex Trax,
  Hewlett-Packard and NYSE:BA head aliases. This selected error is not merely
  a confusion between two near-identical film relations.

The concrete observation is loss of the correct relation decision under both
learned inference modules despite usable support-fitted decisions on the same
queries. This audit inspected query and support center entities, not all sampled
neighbor biographies/attributes, and does not identify a topology mechanism or
prove semantic information was preserved. No annotations were corrected.

Full exact relation paths, IDs, supports and source hashes are retained in
`data/public_crossover_semantic_examples_20260907.json`; reproduce read-only with
`inspect_public_crossover_examples.py` on Tucker. Initial schema checks failed
before interpretation because the capture is nested and centers are list-valued;
the final audit follows the verified saved schema and output order.
