# NM ladder with Facebook inserted at rung 6 (Order D)

Matched-40k, one-hop, 3-shot/30-way NM AUC with within-source balanced sampling.
Order D reuses Order A rungs 1–5, trains new rungs 6–8, and reuses the all-nine
rung-9 model. The 27 new cells are preserved in `data/orderD_new_metrics.csv`; the
assembled trajectory is `data/orderD_ladder_9x9.csv`.

## Result

- Facebook is `0.972567` while held out at rung 5 and `0.995407` when inserted at
  rung 6: `+0.022841` AUC.
- Because the baseline is already close to the ceiling, this is better expressed as
  an **83.3% reduction in pairwise ranking error** (`1 - AUC`: `2.743%` to `0.459%`).
- The five incumbent in-training graphs are effectively unchanged at Facebook entry:
  their mean change from rung 5 to rung 6 is about `+0.0009` AUC.
- Facebook remains `0.993919` at rung 7, `0.993709` at rung 8, and `0.992088` at
  rung 9. It retains `+0.019522` AUC over its pre-entry baseline after all later
  sources are added, still a 71.2% reduction in ranking error.
- The later entry events remain visible: Ukraine-suspended gains about `+0.156` at
  rung 7, TwiBot gains `+0.013` at rung 8, and CP-HK gains `+0.130` at rung 9.

The result strengthens the ladder interpretation: a source-specific entry gain can
remain large near the AUC ceiling, and adding that source does not disrupt incumbent
graphs. The single-seed caveat still applies.
