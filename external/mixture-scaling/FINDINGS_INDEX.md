# Consolidated findings index

This is the canonical entry point for the standalone GraphSAGE and node-MLP
mixture-scaling program. It distinguishes current evidence from historical,
superseded, and recovery-only material. Detailed protocols, tables, receipts, and
limitations remain authoritative in the linked files.

## Main conclusions

1. **Mixture breadth is not monotonically beneficial.** In the strict GraphSAGE
   study, source identity and target compatibility matter more than source count.
   The corrected held-out-edge ladder has no general target-centered slope with
   log mixture size. See [strict GraphSAGE results](results/STRICT_STUDY.md).
2. **Source-pair effects are heterogeneous.** The completed Social-9 lattice shows
   both positive and negative pair effects; target inclusion explains much of the
   apparent benefit. See [GraphSAGE pair composition](results/social9_source_lattice/pair_delta_analysis/FINDINGS.md).
3. **Negative transfer survives without message passing.** Node-only MLP source
   rankings and target-specific interference remain on the GTE graphs. A cumulative
   ladder tracks its best available specialist better than its specialist mean, but
   every final rung remains below the best specialist. See [MLP ladder comparison](results/mlp_transfer_explains_ladder/FINDINGS.md).
4. **MLP mixtures simultaneously rescue and break predictions.** Ukraine+Facebook
   damages more positive-negative orderings than it recovers, whereas Political+
   Suspended does the reverse. Failures concentrate differently by GTE cosine and
   context availability. See the [paired MLP error analysis](scripts/experiments/analysis/transfer/mlp_pair_chat_closeout_20260912/data/mlp-error-analysis/FINDINGS.md).
5. **Input shift is real but not a sufficient explanation.** Neighbor context makes
   graph identity easier to detect, yet measured distribution distance does not
   explain which source pairs transfer well. See the [input-shift audit](results/input_shift_audit/FINDINGS.md).
6. **Some defects are correctable parameterization problems.** On Midterm, adding a
   learned decoder bias lowered held-out BCE from 0.6177 to 0.1078 and increased AUC
   from 0.9552 to 0.9822. See the [decoder test](results/decoder_bias_midterm/FINDINGS.md).
7. **Retention interventions expose a frontier rather than a universal repair.**
   Asynchronous distillation can improve a matched joint-training control, but it
   generally does not recover the stronger singleton across targets; the prescribed
   endpoint-weight grid finds no model clearing both source-validation floors. See
   [asynchronous convergence](results/async_convergence/FINDINGS.md),
   [extension](results/async_extension/FINDINGS.md),
   [KD-weight replication](results/async_kd_weights/FINDINGS.md),
   [seed replication](results/async_seed_replication/FINDINGS.md),
   [matched seed-2 control](results/async_seed2_control/FINDINGS.md), and
   [weight merging](results/async_weight_merge/FINDINGS.md).

## Current evidence map

| Area | Canonical record | Status |
|---|---|---|
| GraphSAGE mixture scaling | [RESULTS](results/RESULTS.md), [strict study](results/STRICT_STUDY.md) | Current, with experiment-local qualifications |
| Social-9 GraphSAGE/GraphMAE lattice | [pair-delta findings](results/social9_source_lattice/pair_delta_analysis/FINDINGS.md) | Complete descriptive single-seed lattice |
| Node-only and fixed-context MLP | [nonzero-mini findings](results/nonzero_mini_transfer/FINDINGS.md), [bias update](results/nonzero_mini_transfer/BIAS_LP.md), [uniform evaluation](results/nonzero_mini_transfer/UNIFORM_EVAL.md) | Consult the newest protocol-specific note before quoting a matrix |
| MLP failure mechanisms | [closeout](scripts/experiments/analysis/transfer/mlp_pair_chat_closeout_20260912/FINDINGS.md) | Consolidated prediction, gradient, and trajectory diagnostics |
| MLP ladder | [diagnostics](results/mlp_ladder_diagnostics_20260911/FINDINGS.md), [specialist comparison](results/mlp_transfer_explains_ladder/FINDINGS.md) | Historical rungs involving the old Suspended artifact remain qualified |
| Preservation audit | [retention index](scripts/experiments/analysis/maintenance/experiment_retention_20260912/FINDINGS.md) | Recovery/archive authority, not a new scientific result |

## Archive boundary

The `experiment_retention_20260912/data/narratives/` tree contains verbatim
worktree snapshots. Those copies are provenance and recovery material, not parallel
canonical findings. Prefer the live records linked above. Dirty-worktree patches and
compressed snapshots in the retention tree were preserved without adjudicating them
as valid results.

Large graph artifacts, checkpoints, W&B runs, and raw private pair diagnostics remain
outside Git under the recorded Tucker paths. Their absence from this repository is
not evidence that an experiment did not run.
