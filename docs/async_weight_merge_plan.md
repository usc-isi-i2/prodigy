# Eleven weight mixtures of the existing seed-2 continuation endpoints

Authorized after the [matched control](../results/async_seed2_control/FINDINGS.md). Status: implemented, pending completion. No new training or seeds. This tests one straight interpolation line between two fixed endpoints, not a full iterative branching/merging training procedure.

Use the existing seed-2 weight-0.1 KD and Ukraine-only checkpoints after exactly 60k added optimizer updates each. They share the same parent checkpoint and initialization/optimizer/sampling state at branch start. Cumulative Ukraine exposure differs (46k versus 76k); this is total-update matching, not exposure or compute matching. Do not substitute the Ukraine-only +30k endpoint because it had better measured transfer.

Interpolate every floating model tensor, including decoder bias, as `theta(alpha) = (1-alpha) * theta_KD + alpha * theta_Ukraine_only`, at alpha 0, 0.1, ..., 1.0. Copy endpoints exactly. Require matching tensor keys/shapes/dtypes and identical nonfloating buffers. Do not average optimizer moments, retrain, or represent these inference-only artifacts as resumable training checkpoints. Their recorded step 0 denotes zero new optimizer updates, not their parents' training history.

Use the original fixed seed-0 graph/context data, source-validation pairs, and hard-label training probes. Verify graph/probe/checkpoint provenance and exact reproduction of endpoint source-validation and probe measurements. Save all 11 candidate weights and measurements.

Eligibility requires both own-singleton source-validation AUC floors from seed 2: Ukraine 0.9896764585971831 and Facebook 0.9531494824886322. Among eligible candidates, select maximum Ukraine AUC, breaking ties by higher Facebook AUC and then lower alpha. Freeze the selection before any new downstream test evaluation. If no candidate qualifies, record `no_feasible_merge` and evaluate no candidate on downstream tests. Do not refine the alpha grid, relax either floor, change endpoints, or select a nearly qualifying candidate after seeing results.

For a qualifying merge only, evaluate the frozen model on the original eight targets and report six-graph transfer separately from source-test retention. Passing source validation does not establish test retention or synergy. If no candidate qualifies, this rules out only this particular 11-point interpolation test; it does not establish an architectural impossibility or failure of other merging methods.

Branch `codex/mlp-async-weight-merge`; local worktree `/tmp/mlp-pair-error-code`; dedicated Tucker worktree `/dataMeR1/phil/gfm/mixture-scaling-async-weight-merge`. Source validation uses GPU 0; conditional test evaluation uses GPUs 0–3. The launcher refuses existing partial output and skips completed roots.

```bash
bash scripts/run_async_weight_merge.sh \
 /dataMeR1/phil/gfm/mixture-scaling/state/async_weight_merge_s2
```

`analysis_data.json` holds metadata, all source measurements and frozen selection. `results/COMPLETE.json` records the selection outcome and test coverage. Local analysis/evidence lives under `results/async_weight_merge/`.
