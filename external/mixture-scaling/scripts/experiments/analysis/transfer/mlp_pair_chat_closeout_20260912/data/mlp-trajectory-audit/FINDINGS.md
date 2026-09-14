# Fresh interleaved trajectory audit
Read-only audit of all 28 non-Election pairs and eight learned-bias, node+neighbor singletons. Each singleton reference is its original source-validation-BCE-selected checkpoint. Source validation uses identical fixed seed-0 positive and negative pairs; these are not the smaller final-transfer evaluation pair sets. Three diagnostic pairs also had both singleton and retained joint checkpoints re-evaluated on both source validation sets; results match the logged self-source AUCs.

## What survives
All scheduled validation reports survive (every 2,000 total updates). Only best.pt, latest.pt, and a duplicate terminal checkpoint survive per run. Therefore the audit can inspect source-validation behavior at every logged step, but cannot newly evaluate missing intermediate models on other graphs without reproducing their weights. It says nothing about steps between logged evaluations.

## Main results
At their originally selected joint checkpoints, own-source validation AUC changes versus each source's selected singleton are:
| Pair | First source delta (pp) | Second source delta (pp) |
|---|---:|---:|
| Ukraine + Facebook | -0.7540 | +0.4931 |
| COVID + Midterm | -0.3284 | +0.2179 |
| Political + Suspended | +1.1274 | +0.3312 |

Ukraine's best logged joint source AUC remains 0.5900 pp below its singleton, and COVID's remains 0.3172 pp below. Alternative selection among logged checkpoints cannot remove these deficits. Joint best source AUCs need not occur at the same step for both sources.

Only 4/28 pairs have a logged checkpoint that matches or exceeds BOTH sources' own selected singleton AUCs. Those same four already qualify at the original selected checkpoint: Political + Facebook, Political + Suspended, Midterm + HK, Suspended + Facebook. At a descriptive 0.1 pp tolerance, 9/28 have some qualifying logged checkpoint. This tolerance sensitivity is not a prevalidated selection rule or a significance threshold.

## Exposure and stopping
| Slower source | Singleton selected updates | Joint total selected updates | Joint updates on this source |
|---|---:|---:|---:|
| Ukraine, with Facebook | 44,000 | 16,000 | 8,000 |
| COVID, with Midterm | 42,000 | 34,000 | 17,000 |

Because both trainers use the same per-source shuffling and 1,024-positive batches (including partial final batches), matching source update counts also matches the cumulative positive-example count. The analysis CSV computes actual positive counts including partial batches. Matching source exposure does NOT match total optimizer steps: joint training takes twice as many total steps.

At the latest exact logged exposure matches:
- Joint Ukraine+Facebook at 20,000 total updates vs Ukraine-only at 10,000: Ukraine AUC difference -0.1462 pp.
- Joint COVID+Midterm at 40,000 total updates vs COVID-only at 20,000: COVID difference -0.1245 pp.
No interpolation is used for unmatched steps. Those deficits are smaller than comparisons to the fully selected singletons, but still negative. This is evidence of an exposure mismatch plus a remaining joint-training difference, not a quantitative causal decomposition.

Ukraine+Facebook stops at 22,000 total updates. Between selected step 16,000 and terminal step 22,000:
- Ukraine BCE improves 0.101550 → 0.099814; source AUC improves 98.2204 → 98.3844.
- Facebook BCE worsens 0.179268 → 0.184350, ending the mean-BCE patience window.
- Facebook terminal source AUC is still 95.7859, versus singleton 95.3845.
Thus average BCE stops training while the slower source is still improving and the other source retains an AUC margin. It does not establish that the slower source would catch up if training continued.

## Interpretation
Changing checkpoint selection within the existing logged trajectories is insufficient for the two harmful diagnostic pairs. Before changing architecture, test whether allowing enough source exposure while monitoring per-source AUC retention can reach a feasible joint checkpoint. Preserve weights at every validation step for that experiment. Political+Suspended already demonstrates source-level gain on both graphs with less exposure than either singleton.

These are exploratory fixed-seed results. Source retention does not guarantee transfer to unseen graphs. Existing downstream test targets have already been repeatedly inspected and are development evidence, not untouched confirmation. No training was run during this audit.

## Artifacts and provenance
- trajectories.json: aggregate validation trajectories, selected/final steps, checkpoint inventory, and singleton edge counts.
- reference0.json through reference2.json: fresh read-only source-validation checks.
- source_comparison.csv: selected and best logged source AUC comparisons.
- trajectory_comparison.csv: complete trajectories, actual cumulative positives, exact exposure/budget-matched deltas.
- pair_retention.csv: same-step retention checks and tolerance sensitivity across all 28 pairs.
- source_trajectories.png/pdf: per-source exposure plots.
- analyze.py: analysis and plotting code.

Code branch: codex/mlp-trajectory-audit, commit dc0fdca.
Local code worktree: /tmp/mlp-pair-error-code.
Tucker code worktree: /dataMeR1/phil/gfm/mixture-scaling-trajectory-audit.
Tucker read-only evaluation outputs: /dataMeR1/phil/gfm/mixture-scaling/results/trajectory_audit_20260912.
