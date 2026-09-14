# MLP Order C link-prediction ladder

## Result

Order C provides strong single-seed evidence that broader multi-graph pretraining improves the node-only MLP's link-prediction transfer. Mean ROC AUC across all nine targets rises at every rung, from 0.560 with Election alone to 0.714 with all nine graphs, a gain of 15.36 AUC points.

This increase combines two effects:

1. **Coverage of unseen targets improves.** For each graph addition, compare the same targets before and after the addition, retaining only targets that remain held out afterward. Of these 28 paired changes, 27 are positive. The mean effect of each addition on the remaining held-out targets ranges from +0.83 to +3.12 AUC points.
2. **Adding the target itself gives a large in-domain jump.** Every target added after rung 1 improves when it enters pretraining. The eight entry gains range from +4.87 to +17.41 points and average +9.61 points.

The ladder therefore supports a coverage account of multi-graph pretraining: adding graphs usually improves transfer even before the evaluation graph is included, and including the target produces an additional, much larger gain.

## Ladder summary

| Rung | Graph added | Mean AUC, all targets | Change from prior rung | Mean effect on targets still held out after addition |
|---:|---|---:|---:|---:|
| 1 | Election | 0.560 | — | — |
| 2 | COVID Political | 0.596 | +3.54 pp | +2.18 pp (7/7 improve) |
| 3 | CP/HK | 0.624 | +2.80 pp | +2.94 pp (6/6 improve) |
| 4 | Suspended | 0.655 | +3.15 pp | +3.12 pp (4/5 improve) |
| 5 | Midterm | 0.668 | +1.34 pp | +0.83 pp (4/4 improve) |
| 6 | TwiBot | 0.686 | +1.78 pp | +2.18 pp (3/3 improve) |
| 7 | UKR/RUS | 0.697 | +1.10 pp | +2.79 pp (2/2 improve) |
| 8 | COVID-19 | 0.706 | +0.92 pp | +1.99 pp (1/1 improves) |
| 9 | Facebook | 0.714 | +0.74 pp | No held-out targets remain |

The all-target increment gets smaller as the ladder grows, but it remains positive at every rung. This is compatible with diminishing returns at the aggregate level. It is not a clean estimate of graph count alone because graph identity, training exposure, and count change together.

## Target-level behavior

Eight of nine targets finish above their Election-only baseline. The gains are large for Facebook (+31.10 pp), Suspended (+20.04), COVID-19 (+19.27), Midterm (+17.13), TwiBot (+16.23), COVID Political (+15.92), UKR/RUS (+14.73), and CP/HK (+7.70). Election declines by 3.91 points as other sources enter the mixture.

The full nine-graph model is not the best rung for every target. Relative to each target's best point on this Order C ladder, rung 9 is lower on eight targets:

| Target | Best rung AUC | Full-mixture AUC | Full-mixture regret |
|---|---:|---:|---:|
| Election | 0.779 (r1) | 0.740 | -3.91 pp |
| COVID Political | 0.783 (r3) | 0.759 | -2.37 pp |
| Suspended | 0.736 (r7) | 0.720 | -1.63 pp |
| COVID-19 | 0.731 (r8) | 0.724 | -0.70 pp |
| CP/HK | 0.573 (r7) | 0.568 | -0.50 pp |
| UKR/RUS | 0.660 (r8) | 0.656 | -0.45 pp |
| TwiBot | 0.719 (r8) | 0.715 | -0.34 pp |
| Midterm | 0.684 (r6) | 0.683 | -0.14 pp |
| Facebook | 0.861 (r9) | 0.861 | 0.00 pp |

Thus, broad pretraining is a strong default for average performance and robustness to unknown deployment targets, but it does not dominate target-specific source selection. Election shows the clearest interference, and COVID Political and Suspended also retain nontrivial specialist advantages.

## Comparison with historical single-source specialists

Historical node-MLP specialists are available for six of the nine targets. On those six, the full mixture averages 0.701 versus 0.726 for the retrospective best singleton, a deficit of 2.47 points. The full mixture trails the best singleton on all six targets, by 0.78 to 4.76 points.

This comparison supports the same practical interpretation as the earlier MLP ladder: mixtures provide coverage and reduce the risk of choosing a poor source, while an oracle that knows the best source for each target remains better. It is only a descriptive cross-run comparison. The historical specialists used shorter selected checkpoints and do not fully match this repaired, convergence-trained Order C protocol.

## Interpretation

The useful claim is not simply that the average rises when the target eventually joins training. The more informative result is that almost every new graph also helps the targets that remain unseen. That pattern is consistent with accumulating reusable feature-space structure across social graphs.

At the same time, the target-specific peaks and Election degradation show that mixture dynamics are not uniformly additive. Additional sources can dilute a representation that is already well aligned with a target. The results fit a **coverage plus interference** model:

- breadth increases the chance that training contains a source whose feature-link relation transfers to the target;
- direct inclusion of the target supplies the strongest alignment signal;
- later graphs can slightly erode an earlier target's optimum;
- the full mixture performs well broadly, but not optimally for every known target.

## Limits

- This is one seed and one source order. The monotone aggregate curve is descriptive, not an uncertainty-qualified scaling law.
- Each rung is trained from scratch to a per-source convergence rule. Larger rungs receive more total optimization updates, so source count and compute are confounded.
- The “targets not yet included” mean changes composition at every rung. The 27/28 statistic avoids that particular problem by comparing the same remaining targets across adjacent rungs, but those comparisons are still dependent and order-specific.
- Entry effects are in-domain gains, not held-out transfer.
- The singleton-oracle comparison uses historical results with a partially different training protocol.

## Recommended next analysis

The cleanest confirmation is to run at least two additional source orders under the same repaired protocol and report three quantities separately: fixed-cohort held-out addition effects, target-entry effects, and regret to the best matching singleton. A fixed-total-update ladder would then distinguish benefits of graph diversity from benefits of additional training compute.

## Provenance

The experiment completed on 2026-09-14. It used seed 0, validation-oriented cosine ROC AUC, convergence stopping, the repaired Suspended graph, and Order C: Election, COVID Political, CP/HK, Suspended, Midterm, TwiBot, UKR/RUS, COVID-19, Facebook. The aggregate contains 81 evaluations: nine independently trained rungs evaluated on nine targets.
