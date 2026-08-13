# NM ladder with a GATv2 background encoder — analysis

This analysis is intentionally separate from `nm_ladder/`. It assembles the 64
GATv2 NM evaluations, refuses incomplete matrices, and compares every cell with
the committed GraphSAGE ladder at `../nm_ladder/data/nm_ladder_full.csv`.

Run from the repository root on Tucker after evaluation:

```bash
python3 scripts/experiments/analysis/transfer/ablations/prodigy_nm/encoder_architecture/nm_ladder_gatv2/analyze_results.py \
  --log-root /dataMeR1/phil/gfm/prodigy-gatladder/log
```

Generated evidence lives in this folder's `data/` directory:

- `nm_ladder_gatv2.csv`: the standalone 8x8 matrix.
- `nm_ladder_backbone_comparison.csv`: 64 paired GATv2-versus-GraphSAGE cells.
- `summary.json`: entry deltas, pre-entry ranges, retention, and registered counts.

Render the dedicated result figures locally with Homebrew Python 3.11:

```bash
/opt/homebrew/bin/python3.11 \
  scripts/experiments/analysis/transfer/ablations/prodigy_nm/encoder_architecture/nm_ladder_gatv2/plot_results.py
```

The script writes PNG and PDF versions under `figures/`:

- `nm_ladder_gatv2_trajectory`: the complete GATv2 staircase, with held-out
  segments, entry markers, and primary entry jumps.
- `nm_ladder_gatv2_backbone_comparison`: paired entry deltas and all 64 matched
  GATv2-versus-GraphSAGE cells.

Do not add a `RESULTS.md` until all 64 cells are present and the raw log paths
have been checked. The fixed evaluation episodes make the backbone comparison
paired, but seed 0 remains one training seed; sub-.02 differences are ambiguous.
