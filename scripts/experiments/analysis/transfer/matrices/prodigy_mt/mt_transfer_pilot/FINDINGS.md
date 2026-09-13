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

## Follow-up architecture pilots

These comparisons use 3-shot evaluation, seed 0, and five graphs (25 cells per
condition). The means are descriptive pilot results, not replicated estimates.

### Task conditioning: additive 8D and FiLM8

Across the complete 75-cell comparison, additive 8D task conditioning was nearly
neutral relative to the unconditioned baseline. Seen accuracy changed by -0.0004
and held-out accuracy by +0.0044; seen ROC-AUC changed by -0.0031 and held-out
ROC-AUC by +0.0025. This is at most a marginal transfer benefit.

FiLM8 shifted performance toward seen tasks rather than improving transfer. It
changed seen accuracy/ROC-AUC by +0.0020/+0.0011 relative to baseline, but changed
held-out accuracy/ROC-AUC by -0.0008/-0.0069. Relative to additive task8, its
held-out accuracy and ROC-AUC were lower by 0.0052 and 0.0093. The pilot does not
support the extra FiLM modulation.

### Variable-way decoding

Variable-way decoding produced the clearest held-out signal among these variants.
Relative to fixed 30-way decoding, mean held-out accuracy rose from 0.5844 to
0.6018 (+0.0174) and held-out ROC-AUC from 0.7076 to 0.7240 (+0.0164), while seen
accuracy and ROC-AUC fell by 0.0091 and 0.0061. The gain was not universal and was
substantially driven by TwiBot, so this is evidence for a seen-to-held-out tradeoff,
not a general improvement claim.

### Support-prototype relation scorer

The relation scorer did not preserve the variable-way gain. Versus variable-way,
seen accuracy was effectively tied (+0.00005), while held-out accuracy fell by
0.0217 and held-out ROC-AUC by 0.0234. It was also worse than fixed 30-way on both
seen and held-out means. This panel provides no evidence for retaining the variant.

### Validation-selected transductive refinement

A single global setting (threshold 0.7, alpha 0.25, one iteration) was selected by
mean validation ROC-AUC across all 25 cells, without consulting test metrics. It
reached held-out test accuracy 0.6022 and ROC-AUC 0.7275, only +0.0004 accuracy and
+0.0035 ROC-AUC over the variable-way baseline. The AUC gain was chiefly from
TwiBot and was slightly negative on UKR-RUS. This is a marginal, heterogeneous
pilot gain rather than robust evidence for transductive improvement.

## Evidence and provenance

- Complete matched table: 90 cells.
- Canonical aggregate: `data/summary.json` in this directory.
- Matrix CSVs and figures in `data/` and `figures/`.
- Follow-up aggregates: `data/task8_summary.json`,
  `data/variable_way_summary.json`, `data/support_relation_summary.json`, and
  `data/transductive_selected_summary.json`.
- Produced by the `prodigy-mtfast` worktree; relevant commits culminate in
  `fc63b7bd` (`results: add heldout mixture matrix row`).
- Follow-up worktree result commits: task8 `0d23c346`, FiLM8 `ed22f765`,
  variable-way `2aaaed18`, support relation `1c2cf2c5`, and transductive
  refinement `4ae7247d`.
- Reconstructed from committed outputs and the associated Codex experiment history
  during the 2026-09-12 storage audit.
