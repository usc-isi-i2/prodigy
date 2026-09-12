# Locked KD-weight replication across training seeds

Authorized 2026-09-12 after the [weight study](async_kd_weights_plan.md). Status: completed, all 96 evaluation cells available, no unavailable selections. Training revision `650a57f0c89dc51586dfdea85e8915e17fb585cf`. See [findings](../results/async_seed_replication/FINDINGS.md). The positive seed-0 result is development evidence. **Seeds 1 and 2 are the fresh training replications; Facebook KD weight stays fixed at 0.1.** Weight 1 and seed-specific singletons are controls, not alternative weights to select on transfer.

## What varies and what stays fixed

Training seed affects fresh model initialization, source-positive permutations and training negative samples. Graph split/context caches, validation pairs, hard-label training probes, and downstream test pairs remain the original seed-0 data. The old `--seed` also controls data caches, so explicit `--data-seed 0 --probe-seed 0` separate those roles. All fixed-probe tensor identities and graph/split receipts must agree across singletons, joint models, seeds 1/2, and development seed 0. Evaluation feature construction also explicitly uses data seed 0.

For each new training seed:

1. Train Ukraine and Facebook singletons from that seed's initialization, with the original minimum-validation-BCE selection and stopping rule: validate every 2k updates, save the strict best BCE, patience 3 against a separate reference requiring improvement >1e-4, count staleness only after step 2500, cap at 100k. Initialization is logged but ineligible. Preserve the original float64 validation-BCE calculation. A cap remains a cap rather than convergence.
2. Run the same fresh joint asynchronous-convergence parent: equal alternating supervision, per-source AUC patience, rewind full state to each task's best joint checkpoint, replace its hard labels with KD, and finish when both tasks converge. Parent LR, architecture, teacher selection, temperature, and detector settings are unchanged. Do not force a convergence order or a seed-0 stopping step.
3. Restore the actual all-converged endpoint model, Adam, samplers and RNGs. Reactivate Ukraine hard BCE and keep that seed's frozen Facebook teacher. Continue two paired branches with Facebook KD weight 0.1 or 1.0 for exactly 60k added alternating optimizer updates. Each gets 30k added Ukraine supervision updates and 30k added Facebook KD updates. Final cumulative exposure can differ across seeds because the parent converges at a seed-dependent step; this is variation in the fixed pipeline, not a duration tuned using downstream scores.

If the parent reaches its safety cap without both tasks converging, record that recipe failure and do not fabricate the specified starting state. Its singletons can still be evaluated. Preserve and report any change in convergence order. A prior-tail exact replay check applies only if that historical tail used the same Ukraine-BCE/Facebook-KD objectives and the identical Facebook teacher; otherwise mark it not applicable. Do not relax numerical identity for an asserted replay.

## Locked source-only selection and evaluation

For each branch, retain the fixed endpoint and separately select maximum Ukraine validation AUC subject to Facebook validation AUC meeting **that seed's selected Facebook singleton**. Use the same 16-point grid of added Ukraine exposure: 0 through 30k in 2k increments. Include the starting checkpoint; exact ties choose earlier added steps. Report start fallback or absence explicitly. The locked treatment is weight 0.1; do not choose between weights after seeing these new tests.

Freeze the two selected singleton checkpoints and four fixed/source-selected continuation checkpoints per seed before downstream evaluation: normally 6 models × 8 graph targets × 2 seeds = 96 cells. Use fixed existing pair sets. Keep source-validation retention, source-test retention and six-non-source-graph transfer separate. Report each seed's transfer difference against both constituent singletons and weight 1; identifying the stronger constituent from its test mean is a retrospective comparison, not a training-selection signal.

Primary replication summaries use seeds 1 and 2 only. Show seed 0 separately as development. Two training seeds do not justify a strong significance claim, and the six graph targets are not six independent training seeds. These are still repeatedly inspected graph targets, so additional training-seed replication is not untouched graph/data confirmation.

## Reproduction and artifacts

Use branch `codex/mlp-async-seed-replication` in its own Tucker worktree and tmux session. Only GPUs 0–3 are used. The launcher runs both parents alongside singleton training, then the four continuation branches, freezes all selections, evaluates, and aggregates.

```bash
bash scripts/run_async_seed_replication.sh \
 /dataMeR1/phil/gfm/mixture-scaling/state/async_seed_replication_s12
```

Partial output roots are refused; completed roots are skipped. Checkpoints remain under each seed's parent/singleton/continuation folders. Source-only histories and manifests are exported as `analysis_data.json`; `results/matrix.csv` and `results/COMPLETE.json` record final evaluation coverage. Analysis code and exported evidence live under `results/async_seed_replication/` in the repository.
