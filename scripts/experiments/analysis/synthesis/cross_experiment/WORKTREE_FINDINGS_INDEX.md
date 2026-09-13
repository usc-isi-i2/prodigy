# Tucker worktree findings index

This index maps old Tucker worktrees to the tracked document that preserves their
scientific conclusions. A worktree is eligible for storage cleanup only after its
entry points to a committed findings file and any uniquely valuable raw artifacts
have been reviewed.

| Worktree | Canonical findings |
|---|---|
| `prodigy-native-matrix` | `synthesis/cross_experiment/native_model_result_matrix/FINDINGS.md`; `evaluation/adaptation_efficiency/FINDINGS.md`; archived pre-selection diagnostic under `evaluation/adaptation_efficiency/archive/` |
| `prodigy-rq1`, `prodigy-rq1-native` | `evaluation/adaptation_efficiency/FINDINGS.md` |
| `prodigy-mtfast` | `transfer/matrices/prodigy_mt/mt_transfer_pilot/FINDINGS.md` |
| `prodigy-unconf`, `prodigy-nmglobal` | `transfer/ablations/prodigy_nm/source_schedule/nm_ladder_sequential_nhop2/FINDINGS.md` |
| `prodigy-archsat`, `prodigy-archsat-eval`, `prodigy-archsat-final`, `prodigy-archsat900` | `transfer/matrices/cross_architecture/icl_arch_matrix/FINDINGS.md` |
| `prodigy-radiusfc`, `prodigy-radiusstrat`, `prodigy-radiusstrat-eval`, `prodigy-radiusauc-eval` | `transfer/ablations/prodigy_nm/center_sampling/nm_all9_radius_finalcore/FINDINGS.md` |

## Boundaries

- This index preserves findings, not every checkpoint or raw log.
- The architecture findings explicitly describe a one-seed, early-budget system
  comparison rather than a converged architecture ranking.
- The radius findings include both the three-seed frozen-test result and the later
  seed-0 10k ROC-AUC follow-up.
- The adaptation document contains the current cross-target-selection result; the
  archived diagnostic is retained only for provenance.
- The August 12 runtime archive has no standalone findings document and is not
  covered by this index; review it separately before deletion.
