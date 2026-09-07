# Schedule effects depend on the readout

Saved-prediction audit completed2026-09-07, code03b28481. Tucker output:
`/dataMeR1/phil/gfm/prodigy-schedule-stage-audit/log/stage_audit_20260907`.
Inputs: complete270-cell `analysis_inputs_20260907v1r1` export in the original
trace-schedule worktree. No new model evaluations or target fitting.

## Deployment baseline correction

Equal mean across four support-competent targets and nine rung/seed cells,
fresh stream, same three schedules per ensemble:

| Method | Accuracy | Macro-F1 | AUC | NLL |
|---|---:|---:|---:|---:|
| Full equal ensemble | .790717 | .784852 | .832255 | .509206 |
| TRACE | .798864 | .793350 | .833275 | .532422 |
| U1 equal ensemble | .809308 | .804097 | .847112 | .570164 |

U1 improves discrimination and accuracy but has worse NLL. Do not claim uniform
dominance. On original stream U1 also beats TRACE in accuracy/F1/AUC. Direct U1
is a necessary baseline for the old schedule-fusion claim, which is not a
superiority result over all available readouts.

## What the full model changes

Fresh, averaged across27 models for each target. Corrected/corrupted are fractions
of all query occurrences, not conditional error rates:

| Target | U1 accuracy | Full accuracy | U1 errors corrected | U1 correct corrupted |
|---|---:|---:|---:|---:|
| Political | .919898 | .892204 | .010827 | .038520 |
| Election | .975984 | .974971 | .000868 | .001881 |
| Facebook | .691587 | .667390 | .058196 | .082393 |
| TwiBot20 | .611473 | .585853 | .076244 | .101864 |

Full accuracy minus U1 accuracy equals correction minus corruption exactly.
This is accounting, not causal mediation. Full inference sometimes repairs U1
errors, so it is not simply redundant; its net effect is harmful on average.

## Training schedule and readout interact

For each target/rung/seed, compare blocked and replay100 against interleaved:
72 paired contrasts on the four supported targets. Changes in full and U1
accuracy have descriptive Spearman rho .29027;25/72 have strictly opposite signs.
Mean absolute changes: full .015557, U1 .014246, and their difference .017280.
That difference is the schedule change in net inference correction, not an
independently estimated causal mediator. No significance claim is attached to
these crossed, dependent contrasts.

Interpretation: a schedule ranking under full inference is not necessarily a
ranking of representation utility. This is a plausible bridge between the
pretraining findings and the readout diagnostics, stronger than assuming that
agreement measures preserved geometry. Existing health–accuracy correlation does
not identify the mechanism because both predictions move and shared errors can
increase agreement.

Remaining publication requirement: show a consequential benefit from explicitly
separating representation learning and learned task inference, beyond merely
using the existing U1 ridge baseline. Differentiable ridge meta-learning is prior
art, not a novel method by itself. The data here motivate that question; they
do not yet answer it or establish a universal best training schedule.
