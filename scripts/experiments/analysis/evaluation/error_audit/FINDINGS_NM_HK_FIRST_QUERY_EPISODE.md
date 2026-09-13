# HK→HK NM: first query and its complete episode

9 September 2026. Canonical test episode `0:0`, first failed query occurrence.
All inputs, representations and logits are cached. New work replays only the
frozen native metagraph on CPU and performs explicit edge ablations. No sampling,
encoding, training or GPU use.

## The query-level explanation

The query subgraph contains 89 real nodes, 96 sampled edges and 18 direct center
neighbors. The benchmark assigns it to class 0. It also has held-out test edges
to candidate classes 27 and 29, so there are three legal answers under the
multi-positive LP definition. The model selects class 7, which is not joined to
the query in the cached training, validation or test views. This case therefore
remains wrong after correcting the evaluation.

The stages disagree about *which* rival looks best:

| Stage | Winning class | Assigned class rank |
|---|---:|---:|
| Raw center cosine | 6 | 6 |
| Raw sampled-neighborhood mean | 14 | 8 |
| Learned pre-metagraph mean cosine | 19 | 5 |
| Native metagraph/readout | 7 | 6 |

The encoder materially changes the geometry rather than passing raw similarity
through. Learned query/support cosine is .643 for the assigned class and .747
for class 7; its strongest individual class-7 supports reach .842 and .849.
The assigned class has the largest sampled-node Jaccard (.1364), narrowly ahead
of class 14 (.1270) and legal class 27 (.1228). Exact identity overlap contains
useful evidence, but the model receives independently encoded subgraphs without
cross-subgraph node IDs.

The final score gap is produced primarily by the positive-support construction
of the class references. In an exact additive decomposition along this query,
the positive-support term is 10.465 for the assigned class and 11.368 for class
7. Class 7 wins 11.239 versus 10.488. Negative-support terms are −.499 and −.646
respectively, so they do not selectively depress the assigned class in this
realized forward pass.

Controlled ablations confirm this local mechanism:

- Removing class 7's three positive messages drops its score from 11.239 to
  6.327 and moves the winner to class 8.
- Removing the assigned class's three positive messages drops its score from
  10.488 to 5.502 and its rank from 6 to 28.
- Removing negative messages into the assigned class lowers its score to 9.398
  and rank to 16; removing all negative messages leaves class 7 winning.
- Masking only adjacency-contradicted negatives changes the winner to class 8,
  still not a legal answer.
- Removing the query's second occurrence from the episode changes the first
  query's logits and all label embeddings by exactly zero. Duplicate-query
  interaction is not the cause in this one-layer architecture.

Thus the immediate cause is supported: **the native encoder and positive-support
class-reference path make an unrelated but semantically coherent class more
aligned with the query than every legal anchor.** Conflicting negative messages
and the duplicate query do not cause this particular forward-pass error. Why HK
training learned that geometry remains unresolved.

## What the episode adds

The episode is difficult but not an isolated collapse: 18/120 queries are correct
(15.0%), at the 29.3rd percentile of the 512 HK episodes. It predicts 26 of 30
classes, so failure is not concentration on one label. Twenty-two of 30 classes
have all four queries wrong; all four assigned-class-0 queries are wrong, with
assigned ranks 6, 9, 10 and 13.

The exact class-reference decomposition generalizes across its 102 errors:

- Positive-support terms favor the wrong winner in 99/102 errors.
- They are the largest pro-error component in 88/102.
- Their gap exceeds the complete final gap in 74/102; other terms partly oppose
  the error in those cases.
- Negative-support terms favor the wrong winner in only 25/102 and have a mean
  rival-minus-true contribution of −.302, versus +1.798 from positive supports.
- Positive-support terms alone solve 14/120 queries; adding negative-support
  terms solves 17/120, close to the native model's 18/120.

These are exact additive descriptions using each realized normalized class
reference. Components share attention and normalization, so dominance is not an
independent causal percentage. The single-query positive-edge removals supply
the controlled intervention evidence.

## Next scaling test

Apply this same checked decomposition to all 512 HK episodes and compare errors
with correct controls, uniquely answerable versus ambiguous queries, persistent
versus episode-sensitive nodes, and native versus foreign checkpoints. The main
question is whether positive-support dominance is a general HK mechanism or a
property of this below-median episode. If it generalizes, the relevant model
change is support-to-reference learning or training coverage; if it does not,
the broader taxonomy must retain multiple mechanisms.

[First-episode aggregate](data/canonical_split/nm_hk_first_episode_reference_decomposition.json).
Private node-level evidence and ablation outputs remain under `/dataMeR1/phil/gfm/error_audit/first_hk_failure_*`.
