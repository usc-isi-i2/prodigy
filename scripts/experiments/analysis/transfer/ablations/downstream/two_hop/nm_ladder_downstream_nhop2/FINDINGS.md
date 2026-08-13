# Fair-two-hop ladder downstream findings

## Question

Do the recent fair-two-hop neighbor-matching ladders show graph-entry effects on
downstream node classification and repaired static link prediction, and do those effects
depend on the training schedule, split, or per-source exposure?

## Protocol

The completed sweep covers 40 logical rows backed by 39 physical encoders. Classification
uses 10-shot evaluation on four labeled graphs. Static LP uses the repaired pair-conditioned
evaluator with shared degree-matched negatives on five graphs. Every encoder is evaluated
with the fair two-hop sampler (`n_hop=2`, fanouts `9,9`, node cap 101). Temporal LP is
excluded because its evaluator remains invalid.

The comparisons are paired on fixed evaluation cases, but all encoders come from one
training seed. Counts below are therefore descriptive paired measurements, not independent
replicates.

## Results

Static LP shows the clearest and most consistent graph-entry signal. Entry improves static
LP in 19 of 21 eligible cases across the five registered trajectories. The largest mean
entry effect occurs for fixed exposure, Order C (`+0.0712` AUC; 5/5 positive). Fixed
exposure, Order A is also uniformly positive (`+0.0153`; 4/4), as is the split-aware
ladder (`+0.0231`; 4/4). Matched-40k and sequential Order A each improve in 3/4 cases.

Classification is weaker and schedule-sensitive. Across 19 entry cases, only 9 are
positive. Mean changes range from `-0.0069` for the split-aware ladder to `+0.0102` for
fixed exposure Order A. There is no comparable universal classification staircase.

| variant | order | classification positive / n | mean delta | static-LP positive / n | mean delta |
|---|---|---:|---:|---:|---:|
| matched 40k | A | 2/4 | +0.0015 | 3/4 | +0.0072 |
| sequential | A | 1/4 | +0.0080 | 3/4 | +0.0504 |
| split-aware | A | 1/4 | -0.0069 | 4/4 | +0.0231 |
| fixed 10k/source | A | 3/4 | +0.0102 | 4/4 | +0.0153 |
| fixed 10k/source | C | 2/3 | +0.0072 | 5/5 | +0.0712 |

## Interpretation

The robust result is task-specific: adding a graph to pretraining usually produces a
positive entry effect for repaired static LP, but not for node classification. This extends
the historical one-hop observation to several controlled two-hop variants and shows that
the static-LP staircase is not tied to one source order or one training schedule.

The experiment does not establish seed-level uncertainty. In particular, the large
sequential and Order-C effects should be treated as hypotheses for replication rather than
population estimates.

## Saved evidence

- `data/downstream_long.csv`: all task/model cells.
- `data/classification_roc_auc.csv` and `data/static_lp_auc.csv`: primary matrices.
- `data/entry_jumps.csv`: before/after entry comparisons.
- `data/paired_to_matched40k.csv`: controlled variant comparisons.
- `data/pair_lp_floors.csv`: static-LP heuristic and raw-feature floors.
- `data/summary.json`: completeness counts and the descriptive summaries quoted above.
- `figures/`: entry, trajectory, and controlled-comparison plots.

The matching run protocol is in `setup/nm_ladder_downstream_nhop2/`.
