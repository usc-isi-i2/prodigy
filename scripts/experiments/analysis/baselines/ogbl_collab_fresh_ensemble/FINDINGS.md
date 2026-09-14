# Fresh-negative two-model ensemble: validation gate failed

Completed September 14, 2026. The best equal-logit pair (seeds 0+1) achieves
68.5607% validation Hits@50, below the validation-selected standalone seed 0 at
68.7155%. Difference: −0.1548 percentage points, or 93 fewer correct positives
among 60,084. All three pairs fail to beat the control. Stop this bounded branch.
No new 2019 scoring or test archive access occurred.

| Candidate | 2018 Hits@50 | Correct positives |
|---|---:|---:|
| Standalone seed 0 (control) | 68.7155% | 41,287 |
| Standalone seed 1 | 67.4073% | 40,501 |
| Standalone seed 2 | 67.2126% | 40,384 |
| Pair 0+1 | 68.5607% | 41,194 |
| Pair 0+2 | 68.5490% | 41,187 |
| Pair 1+2 | 67.8933% | 40,793 |

## Contract and scope

Reuse existing candidate_fresh_v1 selection checkpoints (150/200/250 updates).
Those checkpoints were originally selected using AA fusion on validation; this
experiment does not search for new standalone-optimal checkpoints. Compute equal
means of raw logits in float64; choose maximum validation Hits, ties lowest pair.
The standalone control is selected on validation, not on the previously seen test
results. No AA fusion, score scaling, weight tuning or additional training.

The three checkpoint state dictionaries each contain 496,385 scalars. A two-model
system has 992,770 neural scalars; conservative full-state bound 992,897 including
both original auxiliary budgets and averaging coefficient. No final ensemble was
deployed or evaluated on 2019. This rejects the specified equal-logit ensemble at
these checkpoints, not every possible ensemble or stronger single model.

## Evidence and provenance

Producer revision: 815c126ff7213668e0a2fe29ae49004865282cd6.
Branch: codex/collab-fresh-ensemble. Tucker worktree:
/dataMeR1/phil/gfm/prodigy-fresh-ensemble. Runtime:
/dataMeR1/phil/gfm/ogbl_collab_compact_joint/fresh_ensemble_v1.
CPU only; no training or GPU use. Selection phase measured 0.725 seconds after
imports/tracking startup. Offline W&B run: mtc7t86s; full directory in selection.json.

All six expected validation cells completed. Panel, score, checkpoint and history
hashes passed; archived checkpoint metrics reproduced; official OGB evaluator and
independent strict-50th-negative metric agreed. Local receipt checks all six hit
counts and winner selection. Source predictions were not regenerated with neural
forward passes. No test_started.json, results.json or test_scores.npz was produced.
The initial Git bundle import named a missing branch ref; importing its HEAD fixed
setup before any evaluation. No scientific protocol deviation or rerun occurred.

Prior extensive 2019 exploration remains disclosed, including test-tuned fusion,
test-supervised trees, and official-negative reuse (99,984/100,000 test negatives).
The historical best fresh standalone test score remains 70.1569%; the fresh fused
mean remains 69.7979%. This experiment establishes no HyperFusion win and no
leaderboard acceptance. Any later proposed research should address forward-year
transfer in a stronger single expert, not automatically extend this fusion search.

Reproduction: scripts/experiments/setup/ogbl_collab_fresh_ensemble/README.md.
Machine-readable evidence: data/protocol.json, data/selection.json, and
data/validation_receipt.json.
