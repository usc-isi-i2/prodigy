# Findings: source composition depends on training horizon and evaluation task

## Result

The two crossed evaluations completed without missing cells: 225 physical native-NM
cells at step 100 and 300 physical downstream-classification cells at step 2,500.
Aggregation expands the shared rung-1/rung-9 checkpoints to 243 and 324 logical cells,
respectively. Native NM used 512 fixed episodes on each of nine graph targets;
downstream classification used 128 fixed 2-way/10-shot episodes on each of four graph
targets. All episode fingerprints match the published reference streams.

The mature and early ladders do not have the same shape. Across the 27 logical
order/rung points, the step-100 versus step-2,500 rank correlation is negative for both
native NM (Spearman -0.414) and downstream classification (Spearman -0.193). These are
descriptive correlations: endpoints are shared physical models and the 27 points are
not independent replicates.

## “Mix is max” is horizon-dependent

At 100 steps, the all-nine model is below the best earlier rung in all six task/order
panels, with a gap larger than one percentage point in four. The final-minus-best
accuracy gaps are:

| Task | Order A | Order B | Order C |
|---|---:|---:|---:|
| Native NM, 100 steps | -2.91 pp | -0.40 pp | -2.64 pp |
| Downstream, 100 steps | -1.59 pp | -0.03 pp | -6.43 pp |

At 2,500 steps, rung 9 is the exact maximum only for native-NM order C, but it is within
one percentage point of the maximum in all six task/order panels:

| Task | Order A | Order B | Order C |
|---|---:|---:|---:|
| Native NM, 2,500 steps | -0.02 pp | -0.24 pp | 0.00 pp |
| Downstream, 2,500 steps | -0.80 pp | -0.59 pp | -0.28 pp |

Thus the strict statement “the largest mixture is always the best checkpoint” is still
false. The defensible result is sharper: **with adequate training, the all-source model
becomes competitive with the best subset across tasks and orders; at 100 steps, source
composition is strongly entangled with optimization horizon.**

## ROC-AUC confirms and sharpens the result

The same comparison using ROC-AUC is in `figures/budget_task_ladders_auc.png`. The
all-nine model rises from 0.638 to 0.872 mean native-NM AUC and from 0.561 to 0.756
mean downstream AUC between 100 and 2,500 steps. At 100 steps, rung 9 trails the best
earlier rung in every panel by 4.00–6.95 AUC points for native NM and 1.66–13.82 points
for downstream classification. At 2,500 steps, every rung-9 gap is below one AUC
point; the largest is 0.71 points for downstream order A.

The AUC ladder ranks also change with budget: step-100 versus step-2,500 Spearman
correlation is -0.436 for native NM and -0.076 for downstream classification across
the 27 logical order/rung points. As for accuracy, these are descriptive correlations
over aliased, non-independent points.

Step-2,500 native-NM AUC is recovered from the original fixed-test worker logs at four
decimal places. The recovery covers all physical ladder cells, and the logged values
agree with the available full-precision specialist replay to the expected rounding
tolerance. The other three task/budget cells use full-precision evaluator outputs.

## Exploratory 100 / 2,500 / 40k native-NM overlay

`figures/nm_auc_budget_100_2500_40000.png` places the historical matched-40k NM ladder
beside the two fixed-test budgets. All three curves are averaged over the same eight
targets, excluding Facebook. The 40k curve ends at rung 8.

Only Order A is source-set aligned rung-by-rung. On that order, step 100 is clearly below
both mature curves. Step 2,500 and 40k are close at rung 1 (0.855 versus 0.861) but differ
by 0.046 at rung 8 (0.876 versus 0.922). That residual cannot be assigned to additional
training because 40k uses the legacy shared evaluator and a separate seed-0 campaign.
Orders B and C are included only as context: their historical 40k source sequences differ
from the current nine-source sequences, so vertical comparisons at a rung do not hold the
training source set fixed.

## Existing saturation evidence is directly relevant

The one-hop pretraining-saturation experiment independently evaluated downstream
classification at steps 100, 500, 1,000, 2,000, 10,000, and 40,000. Its all-eight arm
rises from 0.566 mean ROC-AUC at step 100 to 0.766 at step 500, then remains within a
0.013-wide band through 40k (0.761 at 40k). The compute-matched two-hop replication gives
the same pattern: 0.576 at 100, 0.760 at 500, and 0.751 at 40k.

Those values closely bracket the new all-nine downstream endpoints (0.561 at step 100
and 0.756 at step 2,500), despite coming from all-eight checkpoints and an older
evaluation campaign. This is convergent rather than directly pooled evidence. Together,
the experiments suggest that the important transition is between 100 and roughly 500
steps, not a gradual improvement from 2,500 to 40k. The saturation experiment concerns
downstream classification, however; it does not provide a native-NM trajectory. Replaying
its saved 100/500/1,000/... checkpoints on the fixed native-NM streams would test whether
the same early transition governs the pretext task.

## Interpretation and limitation

This turns the apparent inconsistency into the central result. Early source-addition
curves are not reliable estimates of mature transfer value: they can reverse rank and
make a large mixture look harmful when it is primarily under-optimized. At the same
time, mature downstream performance is flatter across rungs than mature native NM, so
the evaluation task still matters.

The comparison is not yet a pure same-trajectory budget ablation. Step-100 and
step-2,500 models came from separate matched training campaigns; the configs differ in
budget plus loader-worker and bookkeeping controls, and step 100 has one training seed.
The decisive next experiment is to evaluate the saved step-100 checkpoints from the
same three final-core training trajectories on both frozen task panels. That removes
training-stream and seed-count ambiguity while keeping every other comparison fixed.
