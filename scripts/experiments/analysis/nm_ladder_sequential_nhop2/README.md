# Sequential two-hop ladder analysis

This folder receives the terminal-checkpoint results for
`setup/nm_ladder_sequential_nhop2/` and pairs them with canonical order A from the
independently run interleaved two-hop control.

Registered outputs:

- `data/nm_ladder_sequential_nhop2.csv`: one wide row per sequential rung;
- `data/nm_ladder_sequential_nhop2_long.csv`: one row per rung/test-graph cell;
- `data/nm_ladder_schedule_comparison_long.csv`: paired sequential versus interleaved;
- `figures/sequential_vs_interleaved_ladder.{png,pdf}`: matched side-by-side ladder
  heatmaps with absolute AUCs and the newest graph outlined at each rung;
- `figures/sequential_minus_interleaved.png`: secondary paired-delta heatmap.

The primary analysis uses only each run's terminal `state_dict_40000.ckpt`. Source-boundary
checkpoints answer a secondary continual-forgetting question and should not be mixed into
the registered terminal comparison.

Interpret paired deltas by role:

- **newcomer**: the graph trained in the final block at that rung;
- **incumbent**: an earlier source that may have been forgotten;
- **heldout**: a graph not yet present in the training prefix.

The sequential and interleaved arms are one seed each. Report exact paired cell deltas,
means, ranges, and win counts; do not present confidence intervals as though the 64 cells
were independent replications.
