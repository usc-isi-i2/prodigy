# Ukraine-only continuation from the existing seed-2 parent

Authorized after the mixed [seed replication](../results/async_seed_replication/FINDINGS.md). This is one targeted diagnostic branch from the already observed failing seed, with no new random seed. Status: implemented, pending completion.

Restore the exact seed-2 parent endpoint at logical 32k: 16k Ukraine supervision and 5k Facebook supervision plus 11k Facebook KD updates. Preserve model weights, AdamW moments/counters, sampling orders/offsets/generators and global RNGs. Use the unchanged `async_extension.train` engine with `arm=ukraine_only`, training seed 2 and data seed 0. Continue Ukraine BCE for 60k updates at LR 0.0005, weight decay 1e-5, clipping norm 1. No Facebook optimizer updates or teacher forwards. Log both source-validation AUC/BCE and original fixed hard-label training probes; save full state every 2k updates. A fixed horizon is intentional, not a convergence claim.

Reuse the existing weight-0.1 KD branch and singleton test reports. The KD choices are already frozen: +60k fixed endpoint and +56k source-selected checkpoint. Do not retrain or reselect KD. The following four Ukraine-only snapshots are prescribed before observing their test results:

| Existing KD checkpoint | Ukraine-only added updates | Matching basis | Cumulative Ukraine exposure, KD / control |
|---|---:|---|---:|
| +60k fixed | 30k | Ukraine supervised exposure | 46k / 46k |
| +60k fixed | 60k | Total optimizer updates | 46k / 76k |
| +56k frozen selected | 28k | Ukraine supervised exposure | 44k / 44k |
| +56k frozen selected | 56k | Total optimizer updates | 44k / 72k |

Total-update matching is not equal compute: KD also performs teacher forwards. Ukraine-exposure matches must have identical Ukraine pair counters and sampler state; initial model/Adam/sampler/RNG state must match exactly. The control must leave Facebook exposure and its sampler unchanged. Existing Facebook exposure in the shared prefix is not zero and must remain in cumulative counts.

A secondary comparison selects the Ukraine-only checkpoint by maximum Ukraine validation AUC subject to the original seed-2 Facebook singleton floor, 0.9531494824886322. Use the original 16-point grid, 0 through 30k added Ukraine updates by 2k, with ties selecting earlier steps. Report initialization fallback or no qualifying checkpoint explicitly. This selected recipe comparison need not match duration. Freeze these choices before new downstream evaluation. Graph/context/probe/validation/test data remain seed 0; verify original graph and evaluation pair provenance for reused reports.

Evaluate only the new control snapshots on the eight original targets. Report six-graph transfer separately from each source's validation and test retention. Keep both fixed matched comparisons and the targeted frozen-selected comparisons; do not promote whichever result looks best. This diagnostic isolates continued Facebook KD from its removal after the common joint prefix. It cannot establish the value of Facebook in that earlier prefix, and it is not an independent untouched-data confirmation.

Branch `codex/mlp-async-seed2-control`, local worktree `/tmp/mlp-pair-error-code`, dedicated Tucker worktree `/dataMeR1/phil/gfm/mixture-scaling-async-seed2-control`. Train on GPU 0; evaluate on GPUs 0–3. No other GPUs are used.

```bash
bash scripts/run_async_seed2_control.sh \
 /dataMeR1/phil/gfm/mixture-scaling/state/async_seed2_control_s2
```

The launcher refuses existing partial output roots and skips completed roots. Source-only histories and manifests are saved in `analysis_data.json`; test metrics and completion receipt in `results/`. Local evidence and analysis live under `results/async_seed2_control/`.
