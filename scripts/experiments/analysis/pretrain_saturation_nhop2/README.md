# Compute-matched two-hop pretrain saturation — analysis

Dedicated analysis for `setup/pretrain_saturation_nhop2/`. It does not read or mutate the
append-only shared benchmark CSVs.

Status: complete. See [`FINDINGS.md`](FINDINGS.md) for the interpretation and the exact
scope of the claim.

`analyze_results.py` requires a complete result matrix and reads:

- classification ROC-AUC directly from this worktree's `log/eval_sat_h2m_*` metric JSONs;
- repaired regression-probe CSVs from `data/reg_probe/`; and
- the committed `n_hop=1` evidence in `analysis/pretrain_saturation/`.

It expands the one verified shared step-0 checkpoint across the three arm curves and
writes:

- `data/pretrain_saturation_nhop2_long.csv` — the standalone h2 evidence table;
- `data/nhop_comparison.csv` — paired cell-level h2 minus h1 differences;
- `data/summary.csv` — arm/task saturation diagnostics;
- `data/regression_floors.csv` — raw-feature floors from the same probe episodes; and
- `figures/nhop_comparison.png` — h1 dashed versus h2 solid, including true step 0.

Completed evidence: three fresh 40k trajectories, 76/76 classification jobs, and four
regression graph passes (40 rows each), with no logged failures.

```bash
/opt/homebrew/bin/python3.11 \
  scripts/experiments/analysis/pretrain_saturation_nhop2/analyze_results.py \
  --log-root /dataMeR1/phil/gfm/prodigy-sat-h2/log
```
