# Fixed-exposure two-hop NM ladder analysis

This analysis consumes the completed Order A + Order C evaluation sweep from
[`../../setup/nm_ladder_fixed_exposure_nhop2/`](../../setup/nm_ladder_fixed_exposure_nhop2/).
The design holds expected exposure at 10,000 NM episodes per active source, so rung
`r` trains for `r × 10,000` total steps with the fair two-hop sampler.

`data/raw_metrics.csv` is a read-only extraction of the 120 Tucker
`metrics_test_step0.json` files. It contains 15 physical model matrices: eight Order-A
models and seven Order-C models. Order C rung 8 reuses the Order-A all-eight checkpoint,
so `analyze_results.py` expands it into both logical order trajectories while retaining
the shared artifact path.

Rebuild the derived tables and figures locally:

```bash
/opt/homebrew/bin/python3.11 \
  scripts/experiments/analysis/nm_ladder_fixed_exposure_nhop2/analyze_results.py
```

Run the integrity tests:

```bash
/opt/homebrew/bin/python3.11 -m pytest -q \
  scripts/experiments/analysis/nm_ladder_fixed_exposure_nhop2/tests
```

The primary metric is NM test ROC-AUC at 30-way/3-shot. See `FINDINGS.md` for the
interpretation and caveats.

The analysis preserves two distinct Order-A comparisons:

- `data/comparison_to_matched40k_h2_orderA.csv` pairs against the committed matched-40k
  fair-two-hop ladder. This is the controlled exposure comparison: source sets, sampler
  tuple, evaluation protocol, and training seed match; only the total-step schedule
  changes. The matched-40k H2 matrix currently covers Order A, so Orders B/C cannot
  enter this direct comparison.
- `data/comparison_to_matched40k_h1_orderA.csv` pairs against the historical matched-40k
  one-hop ladder. This remains valuable as a cross-protocol replication check, but it
  cannot isolate exposure because both budget schedule and context radius differ.
