# Graph ladder — figures & tables

Shared collection point for the **final figures (PDF) and result tables (CSV)** of the
NM graph-ladder experiment, gathered across the several chats that worked on it.

If you're a chat contributing ladder artifacts: keep the *generating script* wherever it
lives (e.g. `scripts/plotting/nm_ladder/`, this experiment folder, etc.), and copy the
*output* PDF/CSV here so everything the experiment produced sits in one place.

## Naming

`nm_ladder_<what>.<ext>` so the source is obvious, e.g.
`nm_ladder_trajectory.pdf`, `nm_ladder_per_step_delta.pdf`, `nm_ladder_full.csv`.

## Contents

_(populate as artifacts land)_

- from this chat (`nm_ladder_fillin`): `nm_ladder_per_step_delta.pdf`, `nm_ladder_full.csv`,
  `nm_ladder_plus_single_source.csv`
- from the plotting chat (`scripts/plotting/nm_ladder`): `nm_ladder_trajectory.pdf`,
  `nm_ladder_means.pdf`, `nm_ladder_deltas.pdf`, `nm_ladder_deltas_lines.pdf`,
  `nm_ladder_vs_similarity.pdf`
- generalist-vs-specialist scatters (this chat): `nm_ladder_generalist_scatter.pdf`
  (home-turf AUC vs breadth, equal scale, y=x = perfect generalist) and
  `nm_ladder_generalist_scatter_regret.pdf` (both axes = Δ-to-best regret, equal scale).
  Combines the ladder rungs with the single-source specialists; scripts
  `plot_generalist_scatter*.py` live in the experiment folder.
- variants from a separate chat session (`make_ladder_figures.py` in the experiment
  folder): `nm_ladder_generalist_scatter_errorbars.pdf` (the same scatter + ±1 s.e.m.
  cross-graph error bars) and `nm_ladder_heatmap.pdf` (8-rung ladder-only AUC staircase
  heatmap). The combined heatmap + regret heatmap it can also make are already covered
  above by `plot_heatmaps.py`, so it does not re-emit them.
