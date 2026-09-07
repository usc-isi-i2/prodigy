# Health-guided source allocation does not improve the primary comparison

7 September 2026. Completed the previously interrupted four-source Facebook
allocation experiment. All 18 newly trained models had finished; stage-2
verification stopped because a shell TSV reader retained a terminal carriage
return in the schedule string. Revision `4cdf1507` strips only that terminator.
All three 2,500-update schedules then passed the existing consumed-source audit.
The final replay completed 27 checkpoints on each of two cached streams.

## Primary result

Fresh-stream means over three training seeds:

| Training rule | Accuracy | AUC | Reported F1 |
|---|---:|---:|---:|
| Uniform interleaving | .6589 | .7155 | .6586 |
| Health-guided interleaving | .6523 | .7111 | .6592 |
| Blocked sequential | .6553 | .7282 | .6565 |
| Replay-100 | .6543 | .7176 | .6567 |
| Naive merged | .6458 | .7112 | .6448 |

F1 is the existing replay's `f1` field, not a newly computed macro-F1.
Health-guided allocation loses .00651 accuracy and approximately .00436 AUC
against uniform interleaving. Seed-wise AUC changes are mixed: positive for
seeds 0 and 1, negative for seed 2. Do not claim reliable improvement.
The original-stream AUC-selected singleton reaches .6882 accuracy/.7611 AUC;
the health-selected singleton reaches .6842/.7509.

The candidate schedule uses original target supports and unlabeled queries,
ranking four source specialists by agreement and allocating 40/30/20/10 percent
of updates. Every final model receives 2,500 updates, but obtaining the pilot
specialists costs additional training. This is a matched final-model budget,
not equal end-to-end compute. This single-target result does not establish
that every adaptive allocation method fails.

## Research decision

Close this allocation hypothesis without tuning weights on these results.
TRACE agreement predicts behavior but this test does not turn that diagnostic
into an improved training protocol. The U1 ensemble also beats TRACE fusion in
the separate singleton panel. Neither result supports presenting TRACE as the
best available deployment method.

The stronger current explanation comes from the separate class-reference
value-path intervention: graph context participates in constructing the target
classifier, and its value pathway can alter discrimination with queries and
attention held fixed. A public multiclass native-protocol replication is more
informative than additional selector variants. The currently running public-KG
training job is separate work; no replication result is asserted here.

## Evidence

The `data/` directory contains the original analysis exports: comparison cells,
summary, per-seed deltas, and protocol. Complete replay metrics and predictions
remain on Tucker at:

`/dataMeR1/phil/gfm/prodigy-trace-health-guided/log/trace_health_guided/final_replay_20260907v1/`

Training verification receipts and the replay completion log are under the same
root's `launch/`. Compact analysis is `analysis_20260907v1/`.
Local worktree: `/private/tmp/prodigy-trace-health-guided`;
branch: `codex/trace-health-guided`. No main-checkout edits were needed.
