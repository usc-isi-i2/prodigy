# Fixed-panel train–validation gap audit

Nine bias-only neighbor models, best and latest checkpoints, no training or checkpoint selection changes. For each graph, uniformly select fixed equal-sized training-positive and validation-positive panels (up to 20,480 each), and sample independent fresh uniform exact-nonedge panels with five negatives per positive. Reuse these panels across checkpoints. As a control, use the identical negative panel for both positive splits; its negative contribution cancels exactly.

## Selected checkpoints

| Graph | Train BCE | Val BCE | Positive contribution to gap | Negative contribution | Train-positive BCE | Val-positive BCE |
|---|---:|---:|---:|---:|---:|---:|
| Ukraine mini | 0.0697 | 0.0770 | 0.0069 | 0.0004 | 0.2662 | 0.3077 |
| COVID mini | 0.0928 | 0.1082 | 0.0167 | -0.0012 | 0.2809 | 0.3811 |
| Midterm | 0.0734 | 0.0991 | 0.0289 | -0.0031 | 0.2008 | 0.3740 |
| COVID Political | 0.0344 | 0.1103 | 0.0771 | -0.0012 | 0.0607 | 0.5230 |
| Election 2020 | 0.0987 | 0.1038 | 0.0064 | -0.0013 | 0.3040 | 0.3426 |
| Suspended | 0.0861 | 0.1600 | 0.0767 | -0.0028 | 0.2059 | 0.6660 |
| TwiBot20 | 0.1072 | 0.1277 | 0.0212 | -0.0007 | 0.3403 | 0.4675 |
| China/HK | 0.0224 | 0.0347 | 0.0117 | 0.0005 | 0.0496 | 0.1199 |
| Facebook | 0.1138 | 0.1789 | 0.0637 | 0.0015 | 0.2891 | 0.6710 |

The large gaps are overwhelmingly positive-edge generalization gaps. Independent fresh negative panels have nearly the same loss; using exactly shared negatives preserves the gap. The model fits seen positive relationships much better than held-out ones. This supports overfitting on positive relationships, rather than negative-sampling noise or cross-graph bias transfer. It does not by itself distinguish literal memorization from graph-structural differences or feature patterns that generalize poorly.

The BCE decomposition is gap = (val_positive_BCE − train_positive_BCE)/6 + 5*(val_negative_BCE − train_negative_BCE)/6. Negative contributions can be slightly negative from independent finite-sample variation. With a shared negative panel, the gap is exactly the positive contribution.

For Facebook, continuing from selected step 6k to final step 12k increases the fixed-panel gap from 0.0652 to 0.1420. Training-positive BCE improves 0.2891→0.2005 while validation-positive BCE worsens 0.6710→1.0473. This is particularly clear evidence that further fitting is harming held-out positive predictions.

These fixed-panel losses are independent diagnostics, not replacements for the validation panels used to select checkpoints or the final test matrix. Differences from plotted minibatch means and original validation estimates are expected. Positive selection and negative panels are deterministic; one draw, no confidence intervals. All 18 checkpoint evaluations were checked for finite scores, disjoint positive panels, recomputed losses and exact additive decomposition. Source revision 0153881; branch codex/lp-gap-audit, local worktree /tmp/mixture-gap-audit. Tucker worktree /dataMeR1/phil/gfm/mixture-scaling-gap-audit, state/gaps_s0.
