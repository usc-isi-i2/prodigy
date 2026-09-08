# Canonical-split paired NM error audit

Corrects the September 7 exploratory audit, which inadvertently used the full
standalone graph with edge splitting disabled. This rerun uses the exact merged
70/15/15 artifact used to train both seed-0, step-2500 specialists. It does not
retrain models or alter graph artifacts.

The runner imports the final-core fixed-grid planning/replay and TrainerFS export
code. It uses 512 episodes per target/split, 30 ways, 3 supports and 4 queries per
anchor. Test identities must match the archived final-core benchmark fingerprints.
Validation uses its separate stored edge view. All anchor/support and anchor/query
pairs must belong to the intended split and be absent from both other splits in
the actual undirected sampler adjacencies. Both models consume clones of the same
materialized CPU batches. All cached input tensors are hashed before/after replay.

Use a dedicated Tucker worktree and a free owned GPU. In the `prodigy` environment:

```bash
CUDA_VISIBLE_DEVICES=1 python scripts/experiments/setup/nm_error_audit_split/run_audit.py \
  --training-state-root /dataMeR1/phil/gfm/worktree-runtime-archive-20260812/prodigy-final-core/files/state/final_core \
  --out-dir /dataMeR1/phil/gfm/error_audit/nm_canonical_split_20260908 --dry-run
```

Remove `--dry-run` for execution. The output directory must not already exist.
Use `--episodes 32 --targets cp_hk` and a different output directory for smoke.
The 512-episode run uses the published test-fingerprint checks; a reduced smoke
is not a benchmark result. `--threads`, `--batch-size` and paths are overrideable.

Raw core-node exports are retained under each model's `core_ids/` directory.
Source-local exports retain core query/anchor/prediction IDs and the reversible
offset. Source blocks must be contiguous and every remapped node in scope.
Per-model and split results, exact edge checks, and fingerprints are recorded in
`protocol_summary.json`. Bios and private prediction files stay on Tucker.

The bio helper in the error-audit analysis folder consumes the generated
`nm_ukr_vs_cp_hk_test_summary.json` path manifest. Aggregate findings/figures belong
in `scripts/experiments/analysis/evaluation/error_audit/`, not this setup folder.
