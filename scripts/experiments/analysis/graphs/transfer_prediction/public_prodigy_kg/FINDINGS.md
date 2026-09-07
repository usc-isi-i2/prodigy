# Public KG support/query context experiment

Completed 2026-09-07; pinned Wiki-trained checkpoint at nominal step 8000
(8001 updates), FB15K237 native 20-way 3-shot 4-query protocol, 500 episodes,
40,000 query occurrences. One checkpoint and one target, not 500 independent
datasets. Native train-mode batch normalization was retained.

## Evidence

Tucker runtime branch `codex/publickg-paired-state`, revision `ec92da2d`,
checkout `/dataMeR1/phil/gfm/prodigy-publickg-paired-state`.
Run `log/publickg_mechanism_fullgraph` has complete execution status and 500
hashed paired episode artifacts. Analysis `log/publickg_mechanism_fullgraph_summary`
completed successfully and includes summary.json, episode_metrics.csv and all
query cases in query_audit.csv. Checkpoint SHA256:
`5e48d5abc0660f7e537d79d6ccd74150aed38eecc8ca90a048adb6dc94aafee7`.
Explicit replay tolerance 1e-4 absolute, zero relative; revised after full-graph
null-forward diagnostics, documented in setup README. This is not bit-exactness.

| Intervention / decoder inputs | Accuracy | Macro-F1 | OVR AUC | NLL |
|---|---:|---:|---:|---:|
| Native | .737950 | .721805 | .974829 | .873646 |
| Support context removed, full | .399250 | .349846 | .902073 | 2.147528 |
| Support removal, changed references only | .443725 | .401487 | .911903 | 1.917581 |
| Support removal, changed queries only | .710550 | .691703 | .972379 | .975945 |
| Query context removed, full | .389950 | .338603 | .897955 | 2.347648 |
| Query removal, changed references only | .713650 | .689969 | .972204 | .956130 |
| Query removal, changed queries only | .444500 | .404026 | .914843 | 1.988233 |
| First-layer support keys transplanted | .737300 | .721346 | .974856 | .871110 |
| First-layer support values transplanted | .738600 | .722256 | .974757 | .873230 |
| First-layer support keys and values transplanted | .737800 | .721750 | .974733 | .872286 |

Metrics are averaged within episodes; class columns are not pooled across tasks.
Full model accuracy exactly matches the separately completed native evaluation.

## Intellectual decision

The strong result is a role-specific computational separation: support context
chiefly changes the class-reference side of the decoder, while query context
chiefly changes the query side. Both kinds of context are useful on this target.
These crossed interventions are not an additive causal mediation decomposition.
They cannot by themselves distinguish normalization coupling from local graph
message-passing effects.

The nominated first-metagraph-layer value-versus-key explanation does **not**
replicate as a useful effect. Value minus key: +.00130 accuracy, +.000910 F1,
-.0000987 AUC, +.002120 NLL (higher NLL is worse). Conditional episode-bootstrap
95% intervals: accuracy [.000175,.002450], F1 [-.000341,.002145],
AUC [-.0001655,-.0000316], NLL [.001154,.003114]. These tiny mixed effects do not
justify a value-repair claim regardless of a nominal interval excluding zero.

Unlike the social one-metagraph-layer experiment, this native architecture has
two metagraph layers. A first-layer-only transplant is not equivalent to
replacing support context throughout the model. This boundary must be explicit;
do not silently rename a later-layer result as confirmation of the first-layer
prediction. No practical new method is established by this experiment.

Possible route-localization question: is the large support-to-reference effect already
present in the pooled support representation and carried through residual or
later-layer paths, rather than the first attention value path? A comparison of
support-pooled-state replacement with layer-specific transplants can discriminate
these routes. This is deferred: it explains architecture but may not establish
actionable benefit.

## Fixed-episode example inspection

Inspected episode 0, choosing the first corruption and first correction rather
than searching for favorable episodes. Query 0 is correctly assigned local class
0 with probability .98616. Support-context removal changes its prediction to
17; changing references alone reproduces that wrong class (true probability
.14357). Query-context removal instead changes its prediction to 12; changing
queries alone reproduces that wrong class (true probability .00498).
Conversely query 8, true class 2 but natively predicted 8, is corrected by
support-reference replacement (true probability .12392 to .62358).
These cases illustrate both benefits and damage, not a semantic explanation of
relation identity. Captures expose center-node indices but not readable entity
names directly; local class numbers must not be interpreted as relation IDs.

## Advisor decision: next test is actionable repair

Do not prioritize more layer tracing. The strongest alternative is that the
separation is largely architectural, and context deletion is an out-of-distribution
perturbation. Test whether the social value intervention's ranking improvement
can be converted into usable decisions using support labels alone.

Use the nominated HK 50k checkpoint and political target. Build genuinely
leave-one-support-out margins: the held-out support label must be removed from
reference construction, not merely excluded from a final regression. Compare
native and value-replacement references with the SAME support-only regularized
logistic calibration. Keep query representations fixed. Include uncalibrated
arms and U1 ridge. No query-label threshold fitting or target-prior oracle.
Fit calibration per episode; map local classes consistently. Record the fact
that support class balance may differ from query prevalence. A new frozen stream
is needed because original/fresh query outcomes have already been inspected.
Freeze numerical regularization and stream construction before running it.

Decision: if repair beats calibrated native AND U1 in accuracy/F1 with compatible
AUC/NLL, investigate breadth. If it beats native but not U1, prefer the simpler
readout interpretation. If it cannot repair F1, close this value-repair direction.
This test is not yet implemented or launched and no successful method is claimed.
