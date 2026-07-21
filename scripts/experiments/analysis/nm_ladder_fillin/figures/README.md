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
- gain-vs-retention (this chat): `nm_ladder_gain_retention.pdf` — one-panel, two-claim
  summary. A high, gently-sagging line = mean AUC of the graphs already in training (they
  hold, ~.92–.96); a diagonal up-arrow at each rung = the newly added graph jumping from
  its out-of-dist level (plotted at the previous rung) up to its in-training level
  (ukr_susp +.16, cp_hk +.14 the biggest; covid/twibot20 ~flat, already transfer). Two
  dispersion-band variants over the in-training graphs: `nm_ladder_gain_retention_std.pdf`
  (mean ± 1 std) and `nm_ladder_gain_retention_minmax.pdf` (min–max; the envelope widens
  at rung 8 as cp_hk joins as the new low). Script `plot_gain_retention.py` in the
  experiment folder (reads `nm_ladder_full.csv` + embedded fallback).
- gap-to-best means (this chat): `nm_ladder_gap_to_best_means.pdf` — the
  `nm_ladder_means.pdf` figure recast in regret terms. Two lines over rungs = mean
  gap-to-best (AUC − best-per-graph, best over all 16 models = ladder rungs +
  single-source specialists) for graphs in-training (blue) vs all 8 (black). The all-8
  line rises from −0.081 toward the frontier as coverage grows, closing the OOD penalty
  to 0 at rung 8; both converge at the residual in-domain regret (−0.020 — the
  generalist tax vs a per-graph specialist). Script `plot_gap_to_best_means.py` in the
  experiment folder.
- gap-to-best gain/retention (this chat): `nm_ladder_gain_retention_gap_minmax.pdf`
  (+ `_gap.pdf` flat range, `_gap_std.pdf`) — the gain-vs-retention figure recast in
  gap-to-best terms (y = 0 is each graph's best over all 16 models). Each newcomer arrow
  rises from its deep out-of-dist gap (−0.08 to −0.19) up to a near-0 in-training gap, so
  every graph reaches ~its own best on entering (residual −0.001 to −0.039, biggest for
  cp_hk); the in-training mean-gap line hugs 0. In the minmax variant the shaded band =
  min–max over ALL 8 test graphs (a wide wedge that stays deep until rung 8, when the last
  hard graph cp_hk finally enters and it collapses); the std/flat variants show the tighter
  in-training spread. Script `plot_gain_retention_gap.py` in the experiment folder.
- delta boxplot (this chat): `nm_ladder_delta_boxplot.pdf` — the ladder analogue of
  `nmss_delta_boxplot.pdf`. x = ladder model (L1→all8, natural order); each column a
  boxplot of that model's AUC gap to the best ladder model on the 8 test graphs (0=best,
  y inverted). Boxes are uncoloured (neutral); each point is a circle coloured by test
  graph. Boxes tighten & rise toward 0 as sources are added; the low `ukr_susp`/`cp_hk`
  circles jump to 0 only once their graph enters (L6/L8). Script
  `plot_ladder_delta_boxplot.py` in the experiment folder.
