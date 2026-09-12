# Fixed-horizon asynchronous KD continuation

Authorized 2026-09-12 after the asynchronous convergence pilot. Status: implemented; completion requires the run's `results/COMPLETE.json` receipt. Parent protocol: [asynchronous convergence](asynchronous_convergence_plan.md).

Question: did source-specific AUC patience stop Ukraine too early, or does continuing Facebook distillation obstruct recovery relative to Ukraine-only continuation?

## Fixed design

Start both branches from the selected parent KD checkpoint at logical 40k, retaining model, AdamW moments/counters, both sampler orders/offsets/generators, and global RNGs. This checkpoint has 20k Ukraine supervised updates and 5k Facebook supervised +15k Facebook KD updates. All prefix work is shared; do not count it twice as branch-specific work. Explicitly reactivate Ukraine hard-label BCE even if the parent terminal state marks both sources converged.

- `kd_extended`: add 60k alternating updates (30k Ukraine hard BCE and 30k Facebook soft-target BCE), reaching 50k Ukraine supervised updates. Keep the original Facebook joint teacher, temperature 1, weight 1, LR 0.0005, clipping, and sampling unchanged. No new patience stopping or rewind. Never reinstate Facebook's hard-label term.
- `ukraine_only`: add 60k Ukraine hard-BCE updates from the exact same state. Save at +30k for matched Ukraine exposure (50k cumulative Ukraine updates), and at +60k for matched optimizer-update count (80k cumulative Ukraine updates). Do not call this last contrast compute-matched: KD performs an additional 30k teacher batch forwards.

Save full training state, separate-source validation AUC/BCE, and the original fixed hard-label training probes every 2k added updates. Record per-source hard-label/KD update and pair counts, teacher forward work, and elapsed time. Original probe files are loaded directly and hashed; graph/split receipts must match the parent.

Validate exact replay at the overlapping old KD checkpoints (logical42k,44k,46k). At the exposure-matched comparison, require identical Ukraine counters and sampler state. A replay mismatch stops the pipeline before downstream evaluation.

## Source validation and declared evaluation

Primary comparisons are the three fixed endpoints: `kd_fixed`, `ukraine_exposure`, `ukraine_updates`.

Separately, choose one checkpoint per branch with maximal Ukraine validation AUC subject to Facebook validation AUC >= its historical selected-singleton validation AUC. Selection candidates include the start, are restricted to <=30k added Ukraine updates and a common 2k-Ukraine-update grid, and break exact AUC ties by the earliest added step. Both branches thus have the same Ukraine-exposure search range/grid. If the start wins, report a start fallback; if no candidate qualifies, report absence. Do not substitute an unconstrained endpoint and call it preserved.

Determine both-source preservation against the two historical selected singleton source-validation AUCs. Audit all logged checkpoints descriptively, but the common-grid selected models come only from the frozen manifest. Historical singleton thresholds come from verified replays on the same source-validation protocol.

Freeze all five checkpoint selections before running the existing eight-target evaluation. Report retention on training-graph validation separately from the six other graph test results. These targets were repeatedly inspected in earlier experiments, so all transfer findings remain exploratory. The new selection settings were motivated by previous results and are not an untouched confirmatory protocol. No new claim of a general gradient mechanism is intended.

## Run and artifacts

Use its own Tucker worktree, branch `codex/mlp-async-extension`, and owned GPUs0–3 only. From that worktree:

```bash
bash scripts/run_async_extension.sh \
  /dataMeR1/phil/gfm/mixture-scaling/state/async_extension_s0 \
  /dataMeR1/phil/gfm/mixture-scaling/state/async_convergence_s0
```

The launcher refuses ambiguous existing outputs and skips a completed root. `analysis_data.json` contains source-only histories, summaries, references and selection manifest. `results/matrix.csv` and `results/COMPLETE.json` are written only after evaluation. Checkpoints remain on Tucker; analysis exports live under `results/async_extension/` in this repository.
