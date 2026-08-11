# PRODIGY × VISION × GILT common-classification analysis

This folder analyzes the registered one-seed, 100-update architecture matrix from
`setup/icl_arch_matrix/`. It is deliberately a low-budget comparison, not a final
architecture leaderboard.

The analysis validates the exact registered 372-cell Cartesian product, source strings,
classification protocol, finite metric ranges, and per-target episode fingerprints before
producing:

- architecture- and target-level ROC-AUC summaries;
- paired architecture differences on identical model/target cells;
- the best-included-specialist composition rule, both pooled and target-demeaned;
- separate residuals for targets represented in versus held out from each mixture; and
- a three-panel paper figure.

Run after the Tucker aggregate completes:

```bash
/opt/homebrew/bin/python3.11 analyze.py \
  --input data/classification_long.csv \
  --output-root .
```

`FINDINGS.md` and committed data/figures are added only after the registered grid passes
aggregation. The claim boundary is one training seed at update 100 on four binary node
classification targets.

Generated outputs are no-clobber by default. Pass `--overwrite` only when intentionally
regenerating an already archived analysis from the same frozen input; `summary.json`
records the input SHA-256 and pinned upstream revisions.
