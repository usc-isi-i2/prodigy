# Controlled gradient-conflict interventions
Completed 48 paired batch trials: two graph pairs × initial/best/final warm-start checkpoint × eight draws. Each trial evaluates ordinary mixed gradients, symmetric two-task conflict projection, and the ordinary actual AdamW parameter delta rescaled to the corrected update norm. 512 positives/source, 1:5 negatives. AdamW moments/counters and model restored separately for each action. Fixed source validation, no downstream test data. Checkpoints read only.

## Findings
- Ukraine + Facebook, initial: no conflict in 8/8 batches. Projection does nothing. Ukraine validation BCE worsens by 0.014435 on average while its AUC improves by 0.030422 pp. This separates calibration/loss damage from ranking damage.
- Ukraine + Facebook, best: conflict in only 1/8 batches. Ordinary mixed updates improve both training losses in 8/8 batches, but mean validation AUC changes are -0.034799 pp (Ukraine) and -0.043795 pp (Facebook). Both validation BCEs worsen in 5/8 batches. Correction barely changes mean AUC effects (-0.034691, -0.043790 pp). This supports local training/validation mismatch, not a conflict-only explanation.
- Suspended + Political, initial: conflict in 8/8 batches. Ordinary mixed steps improve both validation AUCs in 8/8 batches. Correction improves both in only 4/8 and is worse than the norm-matched ordinary step on all 16 source/batch comparisons. Average AUC changes (Suspended, Political): ordinary (+0.188374, +2.276750) pp; corrected (-0.013357, +1.798691) pp; norm-matched ordinary (+0.200854, +2.546186) pp.
- Ukraine + Facebook, final: conflict in 8/8 batches, yet ordinary and corrected mixed steps both improve both validation AUCs in 8/8 batches. Correction has a small local advantage, insufficient evidence for a full-run advantage.

## Interpretation and limits
Raw training-gradient conflict is not sufficient to predict validation harm. Protecting each training objective can suppress a direction beneficial for validation. Evidence favors investigating training/validation generalization mismatch and optimizer trajectory over treating raw conflict removal as the primary fix.

These are single-step local interventions on existing unregularized warm-start trajectories at LR 0.0005, not newly trained conflict-corrected models or the fresh-start mixed-batch pilot. Eight batch draws at fixed checkpoints are not eight independent training seeds. The norm match controls Euclidean parameter-change magnitude, not function-space displacement. Mean metrics should not be interpreted as every batch having the same sign.

Code: branch codex/mlp-conflict-probe, commit ae6c155.
Local code worktree: /tmp/mlp-pair-error-code.
Tucker code worktree: /dataMeR1/phil/gfm/mixture-scaling-conflict-probe.
Tucker results: /dataMeR1/phil/gfm/mixture-scaling/results/conflict_probe_20260912.


## Literature context (2026-09-12)
These local model-specific diagnostics are consistent with existing multi-task optimization/generalization findings. They do not establish a new general mechanism. The next plan prioritizes a supervised-to-distillation loss switch after per-source convergence, distinct from our BCE-plus-ranking penalty. See /tmp/mlp-pair-error-code/docs/asynchronous_convergence_plan.md for primary references, matched-loss checks, and validation/test separation.
