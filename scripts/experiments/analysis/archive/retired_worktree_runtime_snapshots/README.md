# Retired pre-2026-09-07 worktree runtime snapshots

This directory preserves compact ignored runtime evidence from five Tucker
worktrees removed during the 2026-09-12 storage cleanup. Before removal, every
tracked tree was clean and each worktree HEAD was verified to be an ancestor of
main.

Included material consists of JSON/JSONL records, CSV tables, text logs, replay
stream manifests, hydrated examples, and other small diagnostic outputs. Excluded
material consists of `.pt` episode tensors, `.ckpt` checkpoints, extensionless
`state_dict` files, W&B binary runs, Python caches, and the ordinary tracked source
checkout.

| Worktree | Preserved HEAD | Canonical analysis |
|---|---|---|
| `prodigy-local-transfer-contrast` | `c7d307a2` | `graphs/transfer_prediction/local_transfer_contrast/` |
| `prodigy-trace-schedule` | `d29aed68` | `graphs/transfer_prediction/trace_schedule_scaling/` |
| `prodigy-mechanisms` | `3c88bdbf` | `graphs/transfer_prediction/target_performance_mechanisms/` |
| `prodigy-mechanisms-fixedctx` | `faa19170` | `graphs/transfer_prediction/target_performance_mechanisms/` |
| `prodigy-nmi-overnight` | `6260524e` | `transfer/ablations/prodigy_nm/nm_interventions_overnight/` |

These snapshots supplement the canonical findings and aggregate tables. They are
not self-contained experiment archives: reproducing the experiments requires the
source datasets, environment, and appropriate model checkpoints.
