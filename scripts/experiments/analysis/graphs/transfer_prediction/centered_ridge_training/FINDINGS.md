# Centered, scaled direct-readout training: validation status

No downstream result is available yet. This is the strengthened comparator
specified in the paired setup README, not a claimed new method.

## Matched smoke passed, 2026-09-07

Both20-update arms completed at revision `1fa5a048` on Tucker GPU3. The audit at
`fc9a0396` compared them with the original full isolation run's native arms.
Initial model states match exactly, with no added parameter or random draw.
For each of four sources, all five smoke episode payloads match the consumed
prefix of the corresponding full-run source stream. Payloads cover IDs, roles
and sampled context topology, not a bytewise hash of every feature value.

The learned positive scale changed from1/0.07 to14.7665567(blocked) and
14.6168947(interleaved), remaining finite. This verifies an updating scale;
it is not evidence of improved target generalization.

Smoke checkpoint SHA256:

- Blocked: `2fe34e2c233dbf319e83558a13da13d3225c6c9db20abf91120077f5b35bcb4c`
- Interleaved: `f013832ebd3c6e9bc0c10230219b2c19e5265545f4344ad6d9e795b85540290e`

Tucker smoke output:
`/dataMeR1/phil/gfm/prodigy-centered-ridge-training/log/centered_ridge_training_smoke`.
Reference:
`/dataMeR1/phil/gfm/prodigy-encoder-solver-isolation/log/isolation_full_20260907`.

## Full run launched

Revision `fc9a0396`,2,500 updates per schedule, training seed0, GPU3, independent
output `log/centered_ridge_training_full_20260907` under the same Tucker worktree.
The launcher is sequential; process301351/tmux`centered-ridge-full` was verified
live after launch. Recheck actual process/completion state rather than treating
this historical note as proof that training is still running.

The progress-bar native loss/accuracy are diagnostics of an unused native head,
not the optimized predictor's performance. Training uses only the centered ridge
query loss; the separate `train_ridge_loss` and `train_ridge_logit_scale` records
are the relevant optimization diagnostics. No target checkpoint selection.

Full source/initialization audit and cached downstream evaluation remain pending.
