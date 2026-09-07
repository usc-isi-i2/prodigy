# Nominated contrast: reference collapse and initialization sensitivity

7 September 2026. Prediction fixed before any new 2.5k initialization outcomes.
This is one already-designated configuration comparison, not a source sweep.

## Execution update: completed at 9e19e5eb

Tucker worktree `/dataMeR1/phil/gfm/prodigy-label-context-discovery`, detached
`9e19e5eb`, output `log/discovery_20260907`. All 32 original batches / 128
episodes / 3,072 queries completed across four conditions. The producing
training record `member_cpu_training_20260906/job_012` matches the exact
checkpoint directory and confirms training-table initialization. The runner
checks exact active table rows for all eight label nodes in each batch,
weight/input preservation, current-endpoint reproduction, and query invariance
within each initialization. All checks passed. Five label-helper tests pass.

| Initialization | Intact within AUC | Suppressed within AUC | Intact accuracy | Suppressed accuracy |
|---|---:|---:|---:|---:|
| Current | .841073 | .913556 | .723958 | .803385 |
| Training table | .842376 | .914931 | .731120 | .808268 |

Suppression gains are +7.2483 versus +7.2555 within-episode AUC points;
the initialization interaction is +.0072 points, far below the absolute
50k-original interaction of 8.4274 points. Pooled AUC also improves under
both initializations (.820159 to .906490; .823767 to .907201).

This establishes a contrast, not a universal explanation: the original
2.5k ranking and accuracy benefit is not explained by this initialization
mismatch, whereas the large 50k ranking gain and collapse are strongly
initialization-dependent. The 50k model cannot be treated as an unqualified
longer-trained replication of the same mechanism. Different protocols and
one source-target pair prevent isolating training duration as the cause.

Stop the initialization branch after evaluating the two frozen inequalities.
Both pass: suppressed decision disagreement is .0120443 versus the predicted
upper comparison .398112, and absolute AUC interaction is .00007234 versus
.08427373. Intact decision disagreement is .0175781. Prediction-file SHA256:
`92c2a1f42701420e718751e74cbbc0999e10b120e988727d95bedc9241bc824d`.
Do not add table permutations, new checkpoints, or another metric threshold.
Only code was pushed; all results remain private.

## Label-free evidence read now

From all 128 original episodes in each saved K/V run, form the norm of the
difference between the two normalized final class references. No query labels
or predictions enter this measurement.

| Configuration | Intact mean contrast | Suppressed mean contrast | Median suppressed/intact ratio |
|---|---:|---:|---:|
| 2.5k discovery | .751406 | .454122 | .636066 |
| 50k original | .282174 | .027320 | .108705 |

Zero early episodes versus 60 late episodes have a ratio below .1; this is a
descriptive cutoff, not a learned selection threshold. The configurations
differ in protocol, so this is not an isolated effect of training duration.

## Frozen prediction and scope

Repeat only the existing initialization-by-suppression factorial on the exact
original 2.5k discovery batches: current label inputs versus the unchanged
training table, each with intact versus suppressed support context. Use all
128 episodes and the original batch layout; no new samples or readout fitting.

The less collapsed early references predict (1) less decision disagreement
between initializations under suppression than the 50k original value
.3981119792, and (2) a smaller absolute initialization-by-suppression
within-episode AUC interaction than .08427372685. Report both inequalities,
not a favorable choice after outcomes. Also retain the sign of suppression
under table initialization, pooled AUC, accuracy, and prediction proportions.

This is a new-outcome prediction on an already-observed configuration, not a
pristine held-out domain. Passing it would support a contrast between stable
and collapsed references; it would not prove collapse is the causal variable
or explain source ranking. The geometric measure does not use labels, but
the early/late contrast was chosen after prior performance was known.

If either prediction fails, do not search other checkpoints or alter the
geometric definition. Reassess the collapsed-reference explanation. If both
pass, stop the initialization-control branch and decide whether this bounded
interaction can support the paper alongside its public boundary.

Execution completed as recorded above. The runner supported the original batched
layout, checked exact training-table row usage for all label nodes, and reproduced
the saved current-initialization endpoints before outcomes were interpreted.
Private, uncommitted; worktree `.worktrees/role-topology`, branch
`codex/role-topology-interactions`, code `b377f3bc`.
