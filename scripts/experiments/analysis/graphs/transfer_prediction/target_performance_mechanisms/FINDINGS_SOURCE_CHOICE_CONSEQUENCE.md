# Source choice depends on readout, but decision benefit is limited

Read-only retrospective audit, 7 September 2026. Source:
`data/cross_model_matching/cells.csv`, all 290 rows read. No training,
new evaluation, source-data edit, or prospective selection experiment.

Restrict each target to foreign sources (exclude source=target and raw controls),
and compare the source with highest PRODIGY native AUC against the source with
highest PRODIGY prototype AUC. These are observed target-score maxima, not an
available-label selection method. Original and fresh streams were both inspected
previously; neither is an untouched holdout.

| Target | Original native winner | Original prototype winner | Fresh prototype AUC difference using those two original choices |
|---|---|---|---:|
| Political | Ukraine | Ukraine | 0 |
| Election | TwiBot | TwiBot | 0 |
| Facebook | TwiBot | TwiBot | 0 |
| TwiBot | Ukraine | Midterm | +0.026054 |
| Suspension | Political | Midterm | -0.065613 |

Differences evaluate BOTH chosen sources with the prototype on fresh episodes.
The TwiBot original-stream prototype difference is +0.028188 (0.704954 versus
0.676767); the advantage persists in the fresh stream. Suspension reverses:
its original prototype advantage was +0.026428. Thus changing readout changes
a concrete source choice, but the resulting benefit is not consistent across
targets. The independent fresh-stream best source is not used to replace an
original-stream choice in this table.

Recomputing maxima separately within each stream changes the winner in five
of ten target/stream comparisons. Several gaps are small: Election fresh
0.000732, Facebook fresh 0.004608. The pre-existing untrained U prototype is
stronger than the trained foreign-source mean on TwiBot, so this must not imply
that a different pretrained source is the best practical solution.

## Advisor consequence

This supplies a specific example of a changed source-ranking conclusion,
not the missing general decision rule. Report it as a retrospective evaluation
consequence if the paper needs that illustration. Do not call the second stream
a prospective validation, ignore the suspension reversal, or convert ranking
differences into evidence that pretraining caused a representation benefit.
It does not, by itself, close the high-impact significance gap.

Local analysis: `.worktrees/role-topology`, `codex/role-topology-interactions`.
Source CSV preserved unchanged; findings private and uncommitted.
