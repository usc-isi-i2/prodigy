# Lower Facebook distillation weights

Authorized 2026-09-12 after the [extension control](async_extension_plan.md). Status: completed, three new weights plus reused weight1, eight declared checkpoints and64 evaluation cells. [Findings and figures](../results/async_kd_weights/FINDINGS.md). This tests whether lowering Facebook's KD loss weight relaxes the observed retention/fitting tradeoff.

## Fixed design

Prespecified Facebook KD coefficients: **0.1, 0.3, 0.5**, compared with the completed **1.0** extension. All start from the original KD endpoint at 20k Ukraine supervision updates, not from the later source-selected weight-one checkpoint. Restore the identical model, AdamW moments/counters, RNGs, and per-source samplers. Use the same frozen Facebook teacher, temperature1, learning rate0.0005, clipping norm1, and 1:1 alternating source schedule.

Each new arm executes60k additional optimizer updates:30k Ukraine hard-label BCE batches and30k Facebook soft-target BCE batches. Only multiply the Facebook soft BCE by the new coefficient. Never add Facebook hard-label BCE. Keep hard-label training probes unchanged. Scalar loss weighting with shared Adam moments and clipping is not equivalent to proportionally scaling parameter updates. Weight0 would not be the same as Ukraine-only because of update scheduling, Adam momentum and weight decay; no weight0 alternating arm is introduced here.

All arms reach50k cumulative Ukraine supervised updates and50k cumulative Facebook input updates (5k hard +45k KD including the prefix). The completed weight1 arm is reused at its existing60k added-update horizon. Source code changes retain exactly the weight1 loss computation for backward compatibility. New final counters and both-source sampler states must agree with that reference. Compare start/teacher/probe hashes, graph receipts, LR, optimizer policy, schedule, seed and budget before evaluation.

## Frozen source-only selection

Save every2k added optimizer updates. Retain fixed endpoints as the primary weight comparison. Separately select one checkpoint per weight on the same16-point Ukraine-exposure grid:20k through50k cumulative Ukraine updates, spaced every2k Ukraine updates. Maximize Ukraine source-validation AUC subject to Facebook source-validation AUC >= the historical selected Facebook singleton. Include the start as a fallback; break exact within-weight ties by earliest added step. A missing feasible candidate remains missing.

Choose the overall weight/checkpoint using the same constrained Ukraine AUC criterion, then earliest added step, then largest coefficient for exact ties. Freeze the resulting `chosen_run_id` and all per-weight endpoint/selection exports before reading new downstream evaluations. Weight1 remains an eligible validation fallback. Source thresholds are inherited from verified singleton replays, not chosen from test scores.

Evaluate fixed and source-selected checkpoints for all four weights on the existing eight graph targets (up to64 cells). Distinguish training-graph source-validation retention from the six non-source graph test scores. These graphs have repeatedly informed exploratory development; this is not an untouched confirmatory evaluation, and no novel general transfer mechanism is claimed.

## Reproduction

Use branch `codex/mlp-async-kd-weights` in a dedicated Tucker worktree. The launcher trains on owned GPUs0–2 and evaluates on owned GPUs0–3; GPUs4–7 are untouched.

```bash
bash scripts/run_async_kd_weights.sh \
 /dataMeR1/phil/gfm/mixture-scaling/state/async_kd_weights_s0 \
 /dataMeR1/phil/gfm/mixture-scaling/state/async_convergence_s0 \
 /dataMeR1/phil/gfm/mixture-scaling/state/async_extension_s0
```

Existing partial output roots are refused. A completed root is skipped. `analysis_data.json` records source histories, summaries, references and the frozen manifest; `results/COMPLETE.json` records completed downstream evaluation. Checkpoints remain on Tucker. Analysis exports belong under `results/async_kd_weights/`.
