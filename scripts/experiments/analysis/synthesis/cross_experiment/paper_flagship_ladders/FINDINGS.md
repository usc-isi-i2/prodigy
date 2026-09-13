# Findings: matched paper flagship ladders

The audited grid is complete: 1,080 NM cells, 600 classification cells, 432
capacity-comparison NM cells, and 120 cross-task physical models with preserved
provenance.

There is no universal intervention winner. At the final rung, composition leads
NM and objective leads classification, but their margins over the runner-up are
only 0.000024 and 0.000319 ROC-AUC, below the preregistered 0.001 threshold. Across
the entire ladder, objective leads NM by 0.001363 while exposure leads
classification by 0.001354.

The target-safety audit reinforces that qualified conclusion. Objective is the
safest NM arm (worst target effect -0.000590, no material regression), but its
classification worst-target effect is -0.008910 and two targets materially
regress. The capacity comparison likewise does not establish a uniform wide-model
advantage. The supported conclusion is a target-dependent/Pareto tradeoff.

Exact cells, seed summaries, target effects, provenance, and figures are in
`data/` and `figures/`. Three training seeds quantify observed replication, not
episode-sampling uncertainty.
