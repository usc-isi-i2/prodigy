# HK→HK NM: bridge from realized failures to training supervision

9 September 2026. This note connects the canonical HK failure audit to existing
complete consumed-training records and the executable episode construction. It
adds no training, graph encoding, sampling, or GPU work.

## What the canonical model was trained to do

The native checkpoint was trained for 2,500 updates on HK-only, 30-way episodes,
with three supports and four queries per anchor, a one-hop walk for selecting
members, and independently sampled two-hop contexts. Its August 2026 training
predates randomized member roles and used the historical `lowest_sorted`
behavior: unique walk endpoints were sorted by node ID, the first three became
supports, and the next four became queries.

The current executable path makes the supervision explicit. Each support is
joined to all 30 label nodes: `+1` for its sampled anchor and `-1` for the other
29. A query carries no label into the metagraph and cross-entropy requires its
single sampled anchor. Distinct anchors are enforced, but membership is not
disjoint across anchors. Therefore one real node can be a positive example for
one anchor and a negative example or query target for another in the same
episode. Fresh random walks and context sampling vary the evidence; they do not
remove this many-to-many label conflict.

## Direct evidence from consumed training episodes

The existing training audit read all 2,500 consumed batches for each of three
HK seeds: 10,000 episodes and 1.2 million query occurrences per seed. Anchor,
ordered-member, member-set, and saved-state hashes passed. Under the production
lowest-ID/sorted-role policy, the three seeds show:

| Realized training property | HK rate |
|---|---:|
| Query also appears as a differently labelled support | 22.20–22.23% |
| Query identity has multiple query labels | 38.51–38.55% |
| Support identity has multiple support labels | 59.28–59.30% |
| Episodes with a query/support identity conflict | 100% |
| Identity-only query accuracy upper bound | 75.34–75.37% |

The corresponding mean query/support conflict rates are 2.10% for Ukraine and
0.89% for COVID. The HK rate is stable across five 500-update windows
(22.11–22.44%), so it is not a startup accident. One recorded HK episode assigns
the same account as a support for class 13 and as queries for classes 24 and 29.
The three sampled contexts contain 67, 81, and 65 real nodes, so these are not
identical model inputs even though an identity-only rule cannot satisfy them.

These are complete realized streams for matched three-seed training controls
using the canonical policy. They establish what that training recipe repeatedly
presents. They are not a bit-exact reconstruction of every full context consumed
by the separate canonical seed-0 checkpoint.

## Connection to canonical HK test failures

The test data show the same structural pressure at much larger prevalence than
in Ukraine: 46.24% of canonical HK query occurrences have multiple valid
held-out candidate anchors. Correct multi-positive scoring raises accuracy from
17.86% to 23.26%, proving that 3,315 apparent errors were metric artifacts. It
does not solve the task: 21,332 ambiguous occurrences remain wrong even when any
held-out-positive candidate counts.

Among the 25,817 uniquely answerable errors, 60.93% have the query center inside
a rival support subgraph, versus 41.30% of uniquely answerable successes. This is
a correlation with overlap, degree, and repeated sampling, not a causal effect.
The first canonical failure illustrates why: its true class has the greatest
sampled-node overlap, but an unrelated class has the strongest learned positive
support reference. Removing contradicted negative messages does not rescue it.

Across the full benchmark, inference-only masking of training-edge-contradicted
negative messages slightly reduces both assigned and multi-positive accuracy
and loses more successes than it recovers. Thus contradictory negative messages
are not a sufficient immediate inference-time explanation. The already trained
encoder/readout may encode earlier gradients and is coadapted to the original
message pattern.

## Evidence status

**Established observations:** HK training episodes contain extensive conflicting
exclusive labels; the rate is stable and much higher than in two other sources.
HK test episodes also contain extensive multi-anchor ambiguity. Correcting the
metric explains 3,315/50,464 assigned errors (6.57%). The first query and first
episode fail mainly through learned positive-support class references, while the
full metagraph is beneficial on net across HK.

**Controlled intervention:** deleting contradicted negative messages at
inference does not improve the full benchmark. In the earlier member-policy
training control, role shuffling increases query/support conflicts while
improving downstream political classification, and uniform retention can reduce
conflicts without reliably improving that target. Those results reject a scalar
“more identity overlap causes worse transfer” account, although classification
is not the NM endpoint studied here.

**Still a hypothesis:** repeated exclusive supervision on many-to-many HK
neighborhoods teaches the encoder and positive-support reference path to favor
semantic or high-frequency aliases instead of exact held-out adjacency. No
existing experiment isolates this training-gradient mechanism for HK→HK NM.

## Discriminating next test

First scale the exact positive/negative class-reference decomposition from the
first episode to all 512 cached HK episodes and compare errors with preserved
successes. This determines whether positive-reference dominance is population
wide and identifies the uniquely answerable cohort for a training test.

If it generalizes, use a matched-input, matched-initialization training pair:
retain the same centers, members, contexts, optimizer updates, and exclusive
query target, but stop sending a negative support message when training-view
adjacency says that support is also valid for the rival anchor. Measure the
changed per-example gradients before the run. Evaluate on the fixed canonical
episodes with multi-positive top-1 primary and report uniquely answerable
accuracy, recoveries, and lost successes. This isolates support-label conflict
without changing coverage or support selection. A later multi-positive query
loss is a separate intervention because it changes the target definition.

Source evidence: [first query and episode](FINDINGS_NM_HK_FIRST_QUERY_EPISODE.md),
[overlap-aware evaluation](FINDINGS_NM_HK_OVERLAP_AWARE.md), and
[consumed training conflicts](../../graphs/transfer_prediction/target_performance_mechanisms/FINDINGS_CONSUMED_CONFLICTS.md).
Training provenance remains in the original Tucker log at
`/dataMeR1/phil/gfm/worktree-runtime-archive-20260812/prodigy-final-core/files/log/final_core/train/finalcore_ss_cp_hk_s0_20260807.log`.
