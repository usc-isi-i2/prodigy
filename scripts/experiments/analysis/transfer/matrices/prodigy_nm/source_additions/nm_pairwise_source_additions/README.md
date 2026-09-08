# NM pairwise source additions

This analysis measures the effect of adding each second NM pretraining source to
each single-source specialist. It covers all 36 unordered source pairs and all
nine evaluation targets at seed 0 and checkpoint step 2,500.

## Intervention definition

For base source `A`, added source `B`, and evaluation target `T`, the plotted
quantity is the directional intervention

```text
delta(A + B | A, T) = AUC(A + B, T) - AUC(A, T).
```

Each unordered pair therefore appears in two directional views: `A + B - A`
and `A + B - B`. The complete output has 72 directed interventions by nine
targets, or 648 unique delta cells. Positive values mean that adding the second
source improved macro ROC-AUC; negative values mean degradation.

## Figure organization

- The left column is the base specialist's absolute mean AUC across all targets.
- Rows are grouped by base specialist. Base blocks descend by their absolute
  all-target mean AUC.
- Additions within a base block descend by their mean delta on the seven targets
  held out from both sources.
- Evaluation targets are ordered by the mean delta across all eight additions
  to the Ukraine-Russia base, from least helped to most helped. This places the
  Ukraine-Russia target first.
- Solid cell outlines mark the base source's own target. Dashed outlines mark
  the newly added source's target.
- Green is improvement and red is degradation. All deltas are percentage-point
  changes in macro ROC-AUC.

## Inputs and outputs

The specialist baseline is the tracked seed-zero table at
`../../../cross_model/final_core/data/prodigy_final_core/auc/summary/single_source_metrics_long.tsv`.
The pairwise input snapshot is `data/raw/pair_metrics_long.tsv`, copied from the
Tucker production summary:

```text
/dataMeR1/phil/gfm/prodigy-nm-pairs/log/nm_pairwise_finalcore_eval/production/bs32/summary/pair_metrics_long.tsv
```

Committed outputs:

- `data/nm_directional_intervention_deltas_all_targets_long.csv`: full long-form
  data with baseline AUC, pair AUC, delta, target roles, and summary means.
- `data/nm_directional_intervention_deltas_ranked_wide.csv`: the ranked 72-row
  presentation table.
- `figures/nm_directional_intervention_deltas_all_targets.png`: high-resolution
  annotated heatmap.

## Reproduce

From the repository root:

```bash
MPLBACKEND=Agg /opt/homebrew/bin/python3.11 \
  scripts/experiments/analysis/transfer/matrices/prodigy_nm/source_additions/nm_pairwise_source_additions/plot_directional_intervention_matrix.py
```
