# Schedule effects depend on the readout

7 September 2026. Private exploratory analysis of retained predictions; zero
new model forwards. This is a schedule comparison, not a new nine-source
intervention experiment or a source-ranking claim.

Read all ten `target/{original,fresh}.pt` records from Tucker:
`/dataMeR1/phil/gfm/prodigy-trace-schedule/log/trace_schedule_scaling/analysis_inputs_20260907v1r1`.
Each contains 27 models and shared local query labels: three source-set rungs,
three seeds, three schedules. Compared `full_model` and `U1_pre_meta/ridge`
argmax on those same labels. The original study records matched training
examples across schedules and 128 evaluation episodes per record. This audit
does not independently replay the training-order verification.

## Fresh-stream target means across 27 checkpoints

| Target | Full accuracy | Ridge accuracy | Full wins / ties | Ridge errors corrected by full | Ridge-correct decisions lost |
|---|---:|---:|---:|---:|---:|
| Election | .974971 | .975984 | 2 / 17 | .000868 | .001881 |
| Suspended | .500868 | .483941 | 20 / 3 | .111979 | .095052 |
| Bots | .585853 | .611473 | 2 / 0 | .076244 | .101864 |
| Political | .892204 | .919898 | 2 / 0 | .010827 | .038520 |
| Pages | .667390 | .691587 | 0 / 1 | .058196 | .082393 |

The last two columns are fractions of query occurrences, averaged across
models; their difference exactly equals full-minus-ridge accuracy. They are
paired readout bookkeeping, not causal mediation. Suspended is near chance
and reverses its mean readout preference in the original stream; do not use
it as a reliable example of beneficial learned inference. Original-stream
full/ridge means are Election .982928/.983507, suspended .491030/.500579,
bots .600694/.627303, political .876628/.904827, pages .673286/.692310.

## Do schedule changes survive the common readout?

For each target/rung/seed, compare blocked and replay100 separately against
interleaved. Across four non-suspended targets there are 72 contrasts per
stream (not 72 independent training experiments). Spearman correlation of
full-accuracy changes with ridge-accuracy changes is .103 original and .290
fresh. Of contrasts nonzero under both readouts, signs disagree in 28/56
original and 25/57 fresh. Mean absolute full/ridge changes are .01258/.01247
original and .01556/.01425 fresh. Including suspended gives correlations
.217/.344 with 31/69 and 29/72 opposite-sign nonzero contrasts.

These are descriptive, outcome-inspected comparisons with shared reference
arms, targets and seeds; no independence-based p-values or predictive claims.
Near-zero contrasts can change sign easily. We do not turn sign counts into
evidence that all reversals are robust or that ridge measures representation
quality exhaustively.

## Research decision

Reject the simple interpretation that schedule changes merely improve one
general representation quality that every readout preserves. In these saved
results, the direction of full-model schedule effects often differs from the
common ridge readout. This supports investigating representation/inference
compatibility, but does not identify a learned property causing the difference.
It also weakens interpreting full/U1 agreement as a distinct mechanism without
accounting for ridge's stronger baseline accuracy. The apparent health signal
may partly reward agreement with a better classifier.

Do not promote this to a new method or source-transfer law. The independent
contribution review asks for one specific property predicting an affected versus
unaffected/harmed contrast outside discovery, plus an intervention on that
property. It does not require a universal label-free selector. No new training,
calibration, or benchmark search is nominated here.

Fresh input SHA256 values:

- Election: `0cefe8a2b15cb24d77e79ce0b79a5f9ee6bf60a30ff99bcf06b1faada96df798`
- Suspended: `1283cf9060e010cbcaf684bbc0f4684aa5dbd2dcfc9f6738c65e1b69da295671`
- Bots: `c52c35844fa7f914eb4c5d944e9bd679789c2b711e343482a142f596e23c5bde`
- Political: `b1820b2bd676d252a77819297b0d08cfe9c6682677c3db3acd06823401337e10`
- Pages: `b2df71151c233180cca0224c8a53093d4aca3cb82dd041b85a2f770cba7a7015`

Analysis worktree `.worktrees/role-topology`, branch
`codex/role-topology-interactions`, code `30df8cb5`. No changes to the
producing `codex/trace-schedule-scaling` branch or its original findings.
