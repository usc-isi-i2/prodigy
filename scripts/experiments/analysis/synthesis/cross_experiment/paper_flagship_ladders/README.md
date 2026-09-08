# Matched paper flagship ladders

This analysis joins the original seed-0 source-held-out intervention campaign to
the matched seed-1/2 replicas from `setup/paper_three_seed`. It requires complete
fixed evaluation grids before producing any figure:

- 1,080 NM cells: five designs × eight source-count rungs × three training seeds ×
  nine fixed receiver graphs, with 512 episodes per cell;
- 600 downstream classification cells: the same models on five fixed labeled
  targets, with 128 2-way/10-shot episodes per cell.
- 432 secondary capacity cells: standard and wide encoders × eight rungs × three
  seeds × nine NM receivers.

The flagship plot uses the same receiver panel at every rung. Lines are the mean
over matched training seeds, and shaded regions are the observed seed range. The
included-source, future-source, and permanent TwiBot-20 holdout views are retained
in `data/ladder_per_seed.csv` for the retention/extrapolation diagnosis but are not
substituted for the fixed-panel primary curves.
The headline decision uses both the final rung and a normalized trapezoidal area
over the whole eight-rung curve, always calculated inside each training seed.
`data/ladder_area_per_seed.csv` preserves the replication unit and
`data/ladder_area_summary.csv` records paired effects against the baseline.
Target-level endpoint effects are paired within training seed and retained in
`data/target_effects_per_seed.csv`, `data/target_effects_summary.csv`, and
`data/endpoint_target_robustness.csv`. A macro-level winner is rejected if its
worst target regresses from the matched baseline by more than the same 0.001
practical margin; this prevents a mean gain from hiding a material target loss.
The wide-capacity comparison is rendered separately so it does not crowd or
retroactively redefine the five-design primary figure.

Run after the Tucker campaign has completed and its result directories have been
copied locally:

```bash
/opt/homebrew/bin/python3.11 analyze.py \
  --replicate-nm-root /path/to/nm_evaluation/cells \
  --classification /path/to/classification_long.tsv
```

The audit treats all five designs symmetrically. It permits a universal design
claim only if the same arm leads both the endpoint and whole-curve area on both
fixed panels by at least 0.001 ROC-AUC, and no alternative beats its endpoint by
that practical margin in all three seeds for either panel. Otherwise the paper
reports a target-dependent or Pareto result.
