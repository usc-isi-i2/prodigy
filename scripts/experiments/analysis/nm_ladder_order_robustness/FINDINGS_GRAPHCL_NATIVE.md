# SAMGPT native-objective ladder and specialist matrix

## Result

The missing native-objective ladder is now complete: **27 frozen checkpoints × 9
fixed target graphs = 243 GraphCL evaluations**. Each target uses its fixed prompt
slot and one deterministic corruption/edge-drop view that was not used for training
(`10039 + prompt_slot`, versus training's `39 + prompt_slot`). No checkpoint was
updated during evaluation.

Adding a target to the training mixture lowers its own GraphCL BCE loss in **21/24**
measurable entry transitions. The median entry change is **−0.000649** and the mean
is **−0.01457**; the mean is dominated by the large improvements for
`ukr_rus_suspended` in orders A and C. The three increases are tiny: +0.000008,
+0.000056, and +0.000195.

Mean loss over the same fixed nine targets:

| order | rung 1 | rung 9 |
|---|---:|---:|
| A | 0.098898 | 0.000681 |
| B | 0.004293 | 0.000417 |
| C | 0.050800 | 0.000502 |

The trajectory is not monotonic at every intermediate rung. In particular, order B
spikes at rung 4, and order C worsens slightly over rungs 6–8 before dropping at rung
9. The per-target figures show which held-out target drives each change.

## Native-objective max rule

The nine canonical single-source checkpoints were also evaluated on all nine targets,
giving an **81-cell specialist matrix**. For loss, the max-performance rule predicts
each mixture cell with the minimum loss among specialists already present in that
rung.

Across all 243 mixture cells, the rule is **strongly associated but not exact**:

| native metric | Pearson r | median absolute error | MAE | within 0.001 |
|---|---:|---:|---:|---:|
| BCE loss (minimum rule) | 0.917 | 0.000243 | 0.006844 | 73.3% |
| accuracy (maximum rule) | 0.972 | 0.0000067 | 0.000982 | 88.9% |
| probability margin (maximum rule) | 0.949 | 0.000397 | 0.003454 | 61.7% |

Raw BCE is objective-native but unbounded, so it is not a fair primary scale for
comparing max aggregation with arithmetic-mean aggregation. On the bounded
probability margin, the max rule has MAE **0.00345**, versus **0.03568** for the mean
rule: a **90.3% reduction** (10.3× lower error). Its correlation is **0.949**, versus
**0.475** for the mean rule. Accuracy and `exp(-BCE)` give the same qualitative
answer, but accuracy is nearly saturated and `exp(-BCE)` is less directly
interpretable than probability margin.

The loss rule fits broad mixtures much better than early mixtures: rungs 6–9 have
MAE **0.00144** and correlation **0.992**, versus MAE **0.01117** for rungs 1–5.
The maximum cell-level loss error is 0.158. Accuracy gives a tighter fit but is nearly
saturated (the ladder mean is 0.9974). Probability margin is therefore the primary
max-versus-mean comparison; BCE remains a supporting objective-native measurement.

Thus the native objective supports a **late-rung / approximate best-specialist rule**,
not the nearly exact universal rule seen in the earlier held-out TwiBot-20 AUC prefix.
Early mixtures exhibit genuine joint-training effects in both directions: some beat
every available specialist, while others underperform the specialist minimum.

## Execution and claim boundary

- Ladder checkpoints and evaluation: CARC V100, evaluator commit `c8b4396`.
- Specialist checkpoints: the canonical CARC models, transferred byte-for-byte and
  checksum-verified; evaluation on Tucker H100, commit `b8bc122`.
- Training seed: 39; evaluation-view seed base: 10039.
- One fixed unseen GraphCL view per target. These are deterministic paired comparisons,
  not uncertainty estimates over corruption views or training seeds.
- The hardware differs between the ladder and specialist evaluations. The same sparse
  model path and fixed views are used, but small residuals should not be interpreted
  below hardware-level numerical precision without a matched-hardware rerun.

## Evidence

- [`data/samgpt_graphcl_9x3_carc_v100/metrics_long.csv`](data/samgpt_graphcl_9x3_carc_v100/metrics_long.csv)
- [`data/samgpt_graphcl_specialist_matrix_tucker_h100/metrics_long.csv`](data/samgpt_graphcl_specialist_matrix_tucker_h100/metrics_long.csv)
- [`data/samgpt_graphcl_max_rule/cells.csv`](data/samgpt_graphcl_max_rule/cells.csv)
- [`data/samgpt_graphcl_max_rule/summary.csv`](data/samgpt_graphcl_max_rule/summary.csv)
- [`data/samgpt_graphcl_max_rule/rule_comparison_summary.csv`](data/samgpt_graphcl_max_rule/rule_comparison_summary.csv)
- [`figures/samgpt_graphcl_native_probability_margin_order_A.png`](figures/samgpt_graphcl_native_probability_margin_order_A.png)
- [`figures/samgpt_graphcl_native_probability_margin_order_B.png`](figures/samgpt_graphcl_native_probability_margin_order_B.png)
- [`figures/samgpt_graphcl_native_probability_margin_order_C.png`](figures/samgpt_graphcl_native_probability_margin_order_C.png)
- [`figures/samgpt_graphcl_max_vs_mean_probability_margin.png`](figures/samgpt_graphcl_max_vs_mean_probability_margin.png)
- [`figures/samgpt_graphcl_native_ladder_order_A.png`](figures/samgpt_graphcl_native_ladder_order_A.png)
- [`figures/samgpt_graphcl_native_ladder_order_B.png`](figures/samgpt_graphcl_native_ladder_order_B.png)
- [`figures/samgpt_graphcl_native_ladder_order_C.png`](figures/samgpt_graphcl_native_ladder_order_C.png)
- [`figures/samgpt_graphcl_max_rule.png`](figures/samgpt_graphcl_max_rule.png)
