# Frozen support-only boundary repair experiment

Purpose: determine whether the previously nominated support-value intervention's
ranking gains can become useful decisions, not optimize a new target threshold.
One source-selected HK checkpoint at step50000, political target, native one-hop
fanout100/limit2000, three supports per class, 128 episodes. New episode seed
offset200009 (prior inspected streams used0 and100003). No new pretraining.

Compare native, value replacement, U1 ridge, and the same calibrated version of
each. U1 uses normalized embeddings and ridge regularization1, matching the
existing probe. Each calibration holds out one support from each class per fold,
so three balanced folds cover all six supports. Held-out labels are removed from
metagraph relations and replaced by dummy query labels in the model input.
Original target query labels never enter fitting. All original queries remain as
unlabeled examples; the frozen one-layer architecture forbids outgoing query-to-
reference messages and uses saved evaluation BatchNorm statistics.

Per-arm calibration population-standardizes its own six support margins, then
minimizes mean logistic loss +0.5*(slope²+intercept²), with deterministic Newton
optimization. Regularization is fixed at1. The slope is unconstrained; therefore
calibration can reverse ranking when support evidence favors a reversal. Report
slope signs and do not describe this as ranking-preserving temperature scaling.
No oracle target prevalence or class reweighting is supplied. Balanced supports
can mismatch target prevalence; that is a limitation to measure, not correct
using query outcomes. All six comparators use the identical query stream.

Primary operational outcomes: macro-F1 and accuracy against BOTH calibrated
native and U1 (uncalibrated and calibrated); report AUC and NLL alongside them.
If value repair does not beat U1, do not sell it as a superior deployment method.
If F1 cannot be repaired, close the value-repair direction. A single successful
cell would motivate breadth tests, not establish a general framework.

Validation before launch: ten local tests cover calibration/fold primitives and
actual-model fixture paths, including query-label and held-out-label invariance,
input/weight preservation. A two-episode Tucker smoke is still required; it uses
the same fixed rules and must not inform hyperparameter changes. Full results
must use all128 episodes, not selected favorable prefixes.
