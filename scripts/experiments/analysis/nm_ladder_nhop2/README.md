# NM graph ladder at 2 hops — analysis

Analysis for the compute-matched setup in `../../setup/nm_ladder_nhop2/`. Every
2-hop result uses fanouts `9,9`, node limit 101, and one-hop NM positive walks,
matching `pretrain_saturation_nhop2`. No 1-hop result is copied or overwritten.

After the eight canonical Order A models have been evaluated on the eight test
graphs, assemble its complete subset from the same Tucker worktree:

```bash
python3 scripts/experiments/analysis/nm_ladder_nhop2/assemble_results.py \
  --phase A \
  --log-root /dataMeR1/phil/gfm/prodigy-nmlh2/log
```

This writes `*_order_A.csv` files with 8 wide rows, 64 long cells, and 64 paired
1-hop/2-hop cells. It does not accept or emit blank B/C rows.

After all 21 unique 2-hop models have been evaluated, assemble all three orders:

```bash
python3 scripts/experiments/analysis/nm_ladder_nhop2/assemble_results.py \
  --log-root /dataMeR1/phil/gfm/prodigy-nmlh2/log
```

Outputs under `data/`:

- `nm_ladder_nhop2.csv`: 24 order/rung rows × eight test-graph columns.
- `nm_ladder_nhop2_long.csv`: 192 entry-aligned 2-hop cells.
- `nm_ladder_nhop_comparison_long.csv`: cell-wise pairing with the committed
  1-hop order-robustness table, including `delta_h2_minus_h1`.

The assembler requires every selected cell by default and exits nonzero when
anything is missing. `--allow-partial` is for inspection only; do not treat a
partial table as the experiment result.

Generate the paired summary figure locally with Homebrew Python 3.11:

```bash
/opt/homebrew/bin/python3.11 \
  scripts/experiments/analysis/nm_ladder_nhop2/plot_nhop_comparison.py
```

The figure compares mean ladder AUC and all 21 measurable entry jumps between
1-hop and 2-hop. Add `FINDINGS.md` only after the complete result tables and
figures exist; do not pre-write conclusions.
