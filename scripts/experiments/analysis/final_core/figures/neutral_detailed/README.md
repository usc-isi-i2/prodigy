# Neutral, disaggregated final-core figures

This directory complements the summary figures in its parent directory. It is
designed for inspecting the measurements without presenting or annotating a
finding.

The suite follows four rules:

1. only raw architecture-native primary metrics are plotted;
2. training seeds are never averaged together;
3. ladder orders are never averaged together;
4. each ladder target receives its own panel, without target-entry markers,
   favorable regions, effect transformations, or conclusion-oriented text.

## Contents

- `matrix/`: one exact 9 × 9 source/target heatmap per observed training seed;
- `ladder/`: one nine-panel target breakdown per architecture, seed, and order;
- `index.md`: filters, metric, source-row count, and PNG/PDF paths for all 16
  figures.

All comparable PRODIGY panels use fixed shared scales. SAMGPT BCE figures use
fixed shared logarithmic scales because the raw loss spans several orders of
magnitude. The scale choice is descriptive and is stated on the axes.

Regenerate everything from `../../data/results_full_long.tsv` with:

```bash
/opt/homebrew/bin/python3.11 \
  scripts/experiments/analysis/final_core/plot_neutral_detailed.py
```
