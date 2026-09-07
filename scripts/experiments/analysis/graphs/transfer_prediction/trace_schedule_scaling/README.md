# TRACE schedule scaling analysis

Status: complete. See `FINDINGS.md` for the interpretation and `data/` for the
audited result tables and protocol receipts.

This leaf tests the predeclared transition from two-source sequential gains to
larger-mixture forgetting under an order-only intervention. It also tests whether a
bounded 100-episode replay schedule preserves the useful continuity of blocked
training without its larger-mixture forgetting, and whether target-side U1 health
tracks schedule effects.

Primary outcomes are fresh-stream held-out accuracy and ROC-AUC on Election-2020,
Ukraine/Russia suspended, TwiBot-20, COVID-political, and Facebook page-reference. The
original stream is discovery data; it may select a conventional baseline but never
supplies a reported fresh outcome. TRACE schedule routing uses only episode supports,
intermediate predictions, and full-model predictions. The chance-level suspended
target is reported and subject to the fixed `.55` support-competence abstention rule.

The primary scaling interaction is:

`(blocked - interleaved at rung 2) - mean(blocked - interleaved at rungs 3 and 4)`.

Uncertainty for that interaction and for TRACE-health correlations uses a crossed
bootstrap over training seeds and target datasets. The all-target interaction is
primary; an explicitly labeled secondary view excludes the support-incompetent,
chance-level suspended target. Episode streams are fixed across models; they are not
treated as independent checkpoint seeds.

The analysis retains both cell-level paired contrasts and a target/rung table with
three-seed means, spreads, and win counts; cross-target averages never replace the
per-target schedule effects.

TRACE health fusion is the untuned deployment rule: average probabilities among
checkpoints whose final prediction agrees with their U1 support readout, with an equal
average fallback when no checkpoint agrees. Its comparator is a single checkpoint
selected by labeled original-stream AUC for each target/rung/seed. Method uncertainty
uses the same crossed target × seed bootstrap and averages the three repeated rungs
within each resampled cell.

Rebuild the paper-oriented result figure locally with:

```bash
MPLBACKEND=Agg /opt/homebrew/bin/python3.11 \
  scripts/experiments/analysis/graphs/transfer_prediction/trace_schedule_scaling/plot_results.py
```

The figure preserves individual target × seed schedule contrasts in panel a, shows the
fresh-stream health relationship in panel b, and reports target-level plus crossed-macro
fusion gains in panel c.
