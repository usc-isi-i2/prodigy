# Two-hop NM source-schedule comparison

## Result

Balanced interleaving is substantially safer than blocked sequential training at
the eight-source endpoint. Sequential training helps the newest source slightly,
but forgets earlier sources.

Across the complete 8-rung by 8-target NM comparison, sequential minus interleaved
ROC-AUC averaged **-0.0439** (16/64 cells favored sequential). By role:

| Target role | Cells | Mean sequential − interleaved | Sequential wins |
|---|---:|---:|---:|
| Newcomer | 8 | +0.0059 | 7/8 |
| Incumbent | 28 | -0.0697 | 0/28 |
| Held out | 28 | -0.0324 | 9/28 |

At rung 8, mean NM ROC-AUC was 0.9223 for balanced interleaving, 0.8539 for
blocked sequential training, and 0.8633 for the unconfined arm. The corresponding
four-target downstream-classification means were 0.7439, 0.7502, and 0.7579.
Thus the NM retention advantage of interleaving does not translate into a universal
classification ranking; downstream effects remain target- and task-dependent.

## Interpretation

Blocked training creates a plasticity-retention tradeoff: the newly introduced
graph usually benefits, while every measured incumbent cell declines. Interleaving
is the better default for retaining broad NM capability. The unconfined comparison
does not overturn that conclusion and should not be presented as a generally
superior schedule based on one seed.

## Scope and evidence

- One seed per schedule; the 64 source-target cells are dependent measurements.
- Terminal checkpoint comparison at the registered 40,000-step budget.
- Canonical paired data: `data/nm_ladder_schedule_comparison_long.csv`.
- Downstream comparison: `data/nm_ladder_schedule_cls_comparison_long.csv`.
- Produced across `prodigy-unconf`, `prodigy-nmglobal`, and the sequential/interleaved
  ladder worktrees; associated result commit `f44974e4`.
- Cross-checked against the Codex task history during the 2026-09-12 storage audit.
