# Fixed-exposure two-hop NM ladder analysis

This analysis consumes the completed Order A + Order C evaluation sweep from
[`setup/nm_ladder_fixed_exposure_nhop2/`](../../../../../../setup/nm_ladder_fixed_exposure_nhop2/).
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
  scripts/experiments/analysis/transfer/ablations/prodigy_nm/source_exposure/nm_ladder_fixed_exposure_nhop2/analyze_results.py
```

Run the integrity tests:

```bash
/opt/homebrew/bin/python3.11 -m pytest -q \
  scripts/experiments/analysis/transfer/ablations/prodigy_nm/source_exposure/nm_ladder_fixed_exposure_nhop2/tests
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

## Training histories recovered from W&B (2026-09-03)

All 15 completed training runs remain in `eibl-usc/graph-clip`: A1–A8 and
C1–C7, with C8 sharing A8. Their W&B git provenance points to `7c6bae4`,
the fixed-exposure setup commit. Histories end at zero-based steps 9,999–79,999,
matching the prescribed 10k–80k optimizer updates. Crashed attempts are excluded.

- `data/training_runs.csv`: run IDs, direct W&B links, git commits, and history extents.
- `data/training_history_sampled.csv`: 5,000 W&B-sampled observations per model
  (75,000 total), including training loss and accuracy. This is not a full per-step archive.
- `figures/training_curves.{png,pdf}`: means within 1,000-update bins, plotted
  against optimizer updates. These show training-objective behavior, not downstream validation.

The curves show rapid early learning followed by slower improvement. Several arms
still improve near their endpoints, so the histories do not establish convergence
of every rung. Loss levels across different source mixtures are not directly
comparable because the training episodes differ.

Refresh with `MPLBACKEND=Agg /opt/homebrew/bin/python3.11 plot_training_curves.py --fetch`
from this folder; omit `--fetch` to redraw the saved export without network access.
