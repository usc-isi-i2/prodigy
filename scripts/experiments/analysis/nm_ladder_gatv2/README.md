# NM ladder with a GATv2 background encoder — analysis

This analysis is intentionally separate from `nm_ladder/`. It assembles the 64
GATv2 NM evaluations, refuses incomplete matrices, and compares every cell with
the committed GraphSAGE ladder at `../nm_ladder/data/nm_ladder_full.csv`.

Run from the repository root on Tucker after evaluation:

```bash
python3 scripts/experiments/analysis/nm_ladder_gatv2/analyze_results.py \
  --log-root /dataMeR1/phil/gfm/prodigy-gatladder/log
```

Generated evidence lives in this folder's `data/` directory:

- `nm_ladder_gatv2.csv`: the standalone 8x8 matrix.
- `nm_ladder_backbone_comparison.csv`: 64 paired GATv2-versus-GraphSAGE cells.
- `summary.json`: entry deltas, pre-entry ranges, retention, and registered counts.

Do not add a `RESULTS.md` until all 64 cells are present and the raw log paths
have been checked. The fixed evaluation episodes make the backbone comparison
paired, but seed 0 remains one training seed; sub-.02 differences are ambiguous.
