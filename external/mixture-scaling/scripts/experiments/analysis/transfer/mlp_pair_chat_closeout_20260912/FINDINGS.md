# MLP ladder (2): conversation closeout, 2026-09-12

This is the recovery and handoff index for conversation
`01a0942e-e090-7c01-9c2d-053250c25d9b`. The user authorized preserving its work and
removing the experiment worktree, not either main repository.

## What was missing and is now preserved

The active local worktree was clean at `2effe52373bcb56d342c7379ba7d654e23790846`,
but a clean status did not cover analyses left elsewhere in `/tmp`. We rescued
59 files from six directories: error analysis/ranking pilot, gradient diagnostics,
controlled conflict probes, trajectory audit, lower-learning-rate comparisons,
and mixed-batch inspection. Original findings and numerical tables are under
`data/<original-directory>/`; figures are under `figures/`; analysis helpers are
under `code/`. Original file hashes, byte counts and locations are in
`data/rescued_manifest.json`.

These are preserved historical analyses, not new runs. The original FINDINGS
files retain their protocol qualifications. Any assertion in them that an old
checkpoint path "survives" describes the time of that analysis, not this closeout.
CSV shape, JSON parsing, copy hashes, archive checksum and current remote branch
presence were verified. We did not rerun all historical evaluations or validate
every historical numerical interpretation anew.

## Where the experiment evidence lives

Repository: `https://github.com/philippnoah/mixture-scaling`.
All refs below were checked against live GitHub refs; see
`data/verified_remote_refs.json`.

| Work | Durable evidence |
|---|---|
| Sequential, interleaved, Election extension, mixed batches | Branch `codex/mlp-mixed-batch-pilot`, commit `b2d3360`, `scripts/experiments/analysis/transfer/mlp_pair_training_20260912/`: qualified findings, full tables, exported histories, and validation receipt. |
| Error transitions, norm swaps, ranking preservation pilot | This archive: `data/mlp-error-analysis/`, `figures/mlp-error-analysis/`, `code/mlp-error-analysis/`. |
| Gradient and actual AdamW step diagnostics | This archive: `data/mlp-gradient-diagnostic/`, matching figures/code. Producing branch `codex/mlp-gradient-diagnostic`, revision `30e720a`. |
| Norm-matched conflict intervention | This archive: `data/mlp-conflict-probe/`. Producing branch `codex/mlp-conflict-probe`, revision `ae6c155`. |
| Source exposure/stopping trajectory audit | This archive: `data/mlp-trajectory-audit/`, matching figures/code. Producing revision `dc0fdca`. |
| Lower-LR comparison | This archive: `data/mlp-low-lr/`. Tables are preserved snapshots; do not claim a new confirmatory experiment from their presence. |
| Asynchronous convergence | `results/async_convergence/`, branch `codex/mlp-async-convergence`, result commit `91691a4`. |
| Extended source exposure | `results/async_extension/`, branch `codex/mlp-async-extension`, result commit `7083831`. |
| Reduced Facebook KD weight | `results/async_kd_weights/`, branch `codex/mlp-async-kd-weights`, result commit `b0bd58a`. |
| Fresh-seed replication | `results/async_seed_replication/`, branch `codex/mlp-async-seed-replication`, result commit `c57752b`. |
| Matched Ukraine-only continuation | `results/async_seed2_control/`, branch `codex/mlp-async-seed2-control`, result commit `2ef9337`. |
| Weight interpolation | `results/async_weight_merge/`, branch `codex/mlp-async-weight-merge`, result commit `3fc8ecc`. |
| Input distributions and connectivity | `results/input_shift_audit/`, branch `codex/mlp-input-shift-audit`, result commit `2effe52`. |
| Broader historical ladder/feature analysis recovery | Branch `codex/experiment-retention-audit`, commit `79b8d79`, and `results/mlp_ladder_diagnostics_20260911/`. Historical graph-repair/context-leakage caveats apply. |

All six async result families and the input-shift audit are also present in the
current branch. Older campaign material on another branch need not be merged to
remain recoverable. No shared result table was overwritten in this closeout.

## Scientific handoff

- Plain interleaving can help: the archived input audit's fixed-best-constituent
  comparison has positive transfer deltas in 19/28 pairs. This differs from a
  per-target best-singleton oracle. Ukraine–Facebook remains a difficult pair.
- Training-gradient conflict did not reliably identify local validation harm.
  Local update probes do not establish a new general mechanism or full-run gain.
- Weak Facebook KD was the promising mitigation for Ukraine–Facebook; it was not
  a universally dominant recipe. Fresh-seed results were mixed against singleton
  baselines. The matched Ukraine-only continuation showed a retention/transfer
  tradeoff, and none of the 11 prespecified weight merges met both validation floors.
- The input audit found distinguishable p(x), additional graph discrimination
  from neighbor features, and semantic assortativity relative to nonedges.
  Shift magnitude did not explain the sign of transfer. This does not establish
  a change in p(y|x).
- Ranking-repeat source code is preserved on `codex/mlp-ranking-repeats`; a full
  repeat-result export was not located during this closeout. Do not treat code
  presence as complete numerical evidence for those runs.

## Pending proposals — not run

The last proposed next step was to match context availability/neighbor counts,
then shuffle neighbor means within graph/count strata to break center–neighbor
dependence while preserving marginals. After that, compare shared versus
graph-conditioned edge prediction on overlapping feature regions, with matched
training budgets and validation/test separation. These controls were discussed
but not launched. Downstream label-budget adaptation remains an original research
motivation, not a completed experiment in this thread.

## Raw data, checkpoints and changed Tucker paths

The Tucker worktrees and bare mixture-scaling repository had **already been
retired by another cleanup** before this closeout began. We did not remove them.
In particular `/dataMeR1/phil/gfm/mixture-scaling-input-shift` no longer exists.

Surviving runtime results/logs are now under:

`/dataMeR1/phil/gfm/experiment_archives/top_level_retired_20260912/mixture-scaling-runtime/`

The input audit's compact JSON exactly matches the committed JSON. Its eight
raw graph samples and projection archive remain there (about 475 MiB). The error,
gradient, conflict and trajectory result directories also remain. Their inventories
are recorded in `data/tucker_closeout_audit.json`.

We additionally preserved loose local raw diagnostic arrays privately at:

`/dataMeR1/phil/gfm/experiment_archives/mlp_pair_chat_closeout_20260912/private-runtime.tar.gz`

Its transferred SHA-256 matches the local archive. It contains raw score/endpoint
arrays and metric archives; it is recovery material, not a public evidence table.
The archive directory is private. Its inventory and hash are committed in
`data/private_runtime_manifest.json`. Source helpers were committed through Git,
not copied to Tucker in the runtime archive.

**Checkpoint limitation:** the previous `mixture-scaling/state/...` paths are
absent. A bounded search under `/dataMeR1/phil` did not locate the seed-replication,
seed-two control or merge run directories. This closeout cannot certify that
those checkpoint tensors or exact-resume state still exist. They were never
committed to Git, and hashes/configs are not substitutes for the tensors. We did
not delete any checkpoints, caches, graphs, or archived runtime evidence.

## Local retirement

Target: `/private/tmp/mlp-pair-error-code`, branch `codex/mlp-input-shift-audit`.
The only ignored contents were Python bytecode caches. All meaningful checkout
contents and the newly rescued evidence must be committed and verified on GitHub
before `git worktree remove` is run. Keep the branch. Neither the main
`/Users/philipp/projects/gfm/mixture-scaling` checkout nor the main PRODIGY checkout
is a deletion target. The unrelated Tucker recovery tmux job is untouched.

Restore with `git worktree add <new-path> codex/mlp-input-shift-audit` from the
mixture-scaling repository. Historical local file links in this conversation will
stop working after removal; use this GitHub branch and index instead.
