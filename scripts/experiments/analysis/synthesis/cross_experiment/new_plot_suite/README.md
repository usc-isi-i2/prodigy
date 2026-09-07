# New plot suite

Ten exploratory figures requested after reviewing the analysis inventory. The suite
does not invent absent measurements: figures 4–6 are readiness panels and figure 10
is a diagnostic ranking, clearly labelled as such in the panels and manifest.

Run locally with:

```bash
MPLBACKEND=Agg /opt/homebrew/bin/python3.11 plot_suite.py
```

Outputs are written to `figures/` in PNG and PDF. Derived tabular evidence and the
result-status manifest are written to `data/`.

## Interpretation boundary

- Figures 1, 2, 3, 7, 8, and 9 summarize committed measurements.
- Figure 10 prioritizes embedding dimensions for a future intervention; it does not
  measure a permutation effect.
- Figures 4, 5, and 6 specify missing inputs for new checkpoint/prediction exports.
- Figure 9 keeps training-seed and ladder-order spreads separate and excludes the
  within-evaluation episode standard deviation.
