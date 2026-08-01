# NM graph ladder at 2 hops — analysis

Analysis for the matched setup in `../../setup/nm_ladder_nhop2/`. No 1-hop
result is copied or overwritten.

After all 21 unique 2-hop models have been evaluated on the eight test graphs,
assemble from the log directory in the same Tucker worktree that ran evaluation:

```bash
python3 scripts/experiments/analysis/nm_ladder_nhop2/assemble_results.py \
  --log-root /dataMeR1/phil/gfm/prodigy-nmlh2/log
```

Outputs under `data/`:

- `nm_ladder_nhop2.csv`: 24 order/rung rows × eight test-graph columns.
- `nm_ladder_nhop2_long.csv`: 192 entry-aligned 2-hop cells.
- `nm_ladder_nhop_comparison_long.csv`: cell-wise pairing with the committed
  1-hop order-robustness table, including `delta_h2_minus_h1`.

The assembler requires every cell by default and exits nonzero when anything is
missing. `--allow-partial` is for inspection only; do not treat a partial table as
the experiment result.

Generate the paired summary figure locally with Homebrew Python 3.11:

```bash
/opt/homebrew/bin/python3.11 \
  scripts/experiments/analysis/nm_ladder_nhop2/plot_nhop_comparison.py
```

The figure compares mean ladder AUC and all 21 measurable entry jumps between
1-hop and 2-hop. Add `FINDINGS.md` only after the complete result tables and
figures exist; do not pre-write conclusions.
