# Findings: paper mechanism sweeps

The audited sweep is complete: 810 NM cells, 375 classification cells, 90 physical
checkpoints, and 75 cross-task models with preserved provenance.

There is no task-independent graph-mixing optimum. Classification selects
`p=0.25` (mean AUC 0.767108; runner-up margin 0.003349), but its worst target
effect versus `p=0` is -0.008285. NM selects `p=1.0` (mean AUC 0.868878; margin
0.002594), with a worst-target effect of -0.001033. The winners differ and neither
clears the target-safety condition.

From 2k to 10k steps, NM improves by 0.004570 in all three seeds but its worst
target changes by -0.005329. Classification improves by only 0.000853, is positive
in two seeds, and has a -0.017441 worst-target effect. The wide NM encoder improves
the macro average by 0.004466 in all seeds, but its worst target changes by
-0.001334. None clears every preregistered gate, supporting task- and
target-dependent mechanisms rather than a universal prescription.
