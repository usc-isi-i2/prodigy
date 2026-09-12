# Matched multitask-transfer pilot

## Result

Adding neighbor matching (NM) to multitask (MT) pretraining improved the held-out
panel relative to either objective alone, but did not improve the pooled matrix
relative to NM alone.

| Evaluation | MT | NM | NM+MT |
|---|---:|---:|---:|
| All cells, mean accuracy | 0.5125 | **0.5582** | 0.5516 |
| All cells, mean ROC-AUC | 0.6157 | **0.6911** | 0.6795 |
| Held-out, mean accuracy | 0.5527 | 0.5586 | **0.5844** |
| Held-out, mean ROC-AUC | 0.6682 | 0.6833 | **0.7076** |

Across all cells, NM+MT exceeded MT by 0.0390 accuracy and 0.0638 ROC-AUC,
but trailed NM by 0.0067 accuracy and 0.0115 ROC-AUC. NM+MT beat NM in 12
cells while NM beat NM+MT in 13. The difference was strongly role-dependent:
NM+MT improved the diagonal by 0.1057 accuracy but reduced off-diagonal
accuracy by 0.0348.

## Interpretation

The combined objective is not a universal replacement for NM. Its value is
concentrated in held-out transfer and same-source behavior, while the aggregate
source-target matrix favors NM alone. This is evidence for objective interactions,
not a claim that multitask pretraining monotonically improves transfer.

## Evidence and provenance

- Complete matched table: 90 cells.
- Canonical aggregate: `data/summary.json` in this directory.
- Matrix CSVs and figures in `data/` and `figures/`.
- Produced by the `prodigy-mtfast` worktree; relevant commits culminate in
  `fc63b7bd` (`results: add heldout mixture matrix row`).
- Reconstructed from committed outputs and the associated Codex experiment history
  during the 2026-09-12 storage audit.
