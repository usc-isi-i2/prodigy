# Public boundary: useful context in both roles, with cross-role coupling

7 September 2026. Private analysis; no new model forwards. The completed
fixed public test remains negative for the predicted role-conflict conjunction.

## Scientific result

Original-style Wiki-to-FB15K-237 classification at the nominated 8,001-update
checkpoint: native accuracy 73.795%, raw prototypes 72.4125%, untrained
prototypes 69.8025%. Both paired quality comparisons pass. Suppressing support
edges reduces accuracy to 39.925%; suppressing query edges to 38.995%.
The corresponding paired differences are -33.87 points (95% CI
[-34.5501, -33.1925]) and -34.80 [-35.475, -34.14]. These are conditional
500-episode intervals, not variability across training seeds or graphs.

The hypothesis that support suppression helps while query suppression hurts
does not generalize to this setting. No checkpoint or setting replacement is
authorized by this result. The original failed logit-tolerance verification is
preserved; the explicit accuracy-equivalence amendment preceded baseline and
role outcomes. All native replay labels and 40,000 predictions match.

## What the already-saved internal states establish

Read all 1,500 state files from the three 500-episode streams, verifying each
file's SHA256 against its index and each index against `roles.json`. Check
matching point labels and role masks. Compute per-point L2 change divided by
the intact vector norm, cosine agreement, and norm ratio. Average points
within each episode, then average the 500 episode means. No fitting,
classification, additional intervention or model forward is involved.

| Intervention | Observed vectors / stage | Relative L2 change | Cosine with intact | Norm ratio |
|---|---|---:|---:|---:|
| Suppress supports | Supports, pre-metagraph | 0.50464 | 0.87042 | 0.98790 |
| Suppress supports | Queries, pre-metagraph | 0.14171 | 0.99569 | 1.10105 |
| Suppress supports | Queries, final metagraph | 0.23861 | 0.97220 | 1.01820 |
| Suppress queries | Queries, pre-metagraph | 0.49936 | 0.87568 | 1.00633 |
| Suppress queries | Supports, pre-metagraph | 0.20965 | 0.99174 | 1.15547 |
| Suppress queries | Supports, final metagraph | 0.26054 | 0.96631 | 1.01257 |

Support suppression changes queries **before** either metagraph layer; query
suppression likewise changes supports. Thus public cross-role changes cannot
be attributed solely to messages through class references. Shared training-mode
BatchNorm is a plausible computational source of this coupling, consistent
with the recorded architecture. This observation does not isolate an individual
normalization operation or show that coupling causes the accuracy loss.

Directly intervened pre-metagraph vectors retain nearly unchanged mean norms
but lose directional agreement (cosines about .87). Cross-role vectors change
more in scale and less in direction (cosines above .99). These are descriptive
signatures, not a validated sign predictor and not norm/direction interventions.

## Advisor interpretation

### Post-hoc final-readout decomposition

Using the same 1,500 hash-verified saved states, recombine intact (A) and
intervened (B) final query vectors and final class references. Compute cosine
argmax in float64; native and joint accuracies reconcile exactly with all 500
saved per-episode accuracies. There are zero new model forwards. These mixed
states need not be realizable upstream interventions, and the class references
already include any shared-BN/metagraph effects.

| Edge intervention | A queries / A references | A queries / B references | B queries / A references | B queries / B references |
|---|---:|---:|---:|---:|
| Support suppression | 73.795% | 44.3725% | 71.055% | 39.925% |
| Query suppression | 73.795% | 71.365% | 44.450% | 38.995% |

The support intervention's much larger isolated readout loss follows changed
class references (-29.4225 points) rather than changed queries (-2.74 points).
The query intervention's much larger isolated loss follows changed queries
(-29.345 points) rather than changed references (-2.43 points). The accuracy
interaction `BB - AB - BA + AA` is -1.7075 and -3.025 points, respectively;
these quantities are not additive causal fractions or mediation estimates.

Thus the public boundary rejects opposite utility signs, but its final readout
still shows the distinct reference-versus-query roles. This strengthens the
functional-role interpretation without validating the social support-value
mechanism across tasks or explaining the sign of transfer. Cross-role drift
alone is not an adequate explanation of the large observed loss: the hybrid
readouts indicate that the directly associated final component carries the
larger isolated loss. That remains an endpoint description, not a proof about
the intervening computation.

Keep both conclusions: the public end-to-end role prediction failed, and its
intervention is not the fixed-query causal isolation established in the social
single-metagraph setting. Do not use the coupling observation to erase the
negative result or claim BatchNorm explains the failure. The unresolved issue
is a computational condition that predicts when support-value processing harms
class-reference discrimination. No new experimental block is nominated here.

Evidence: Tucker-only `prodigy-publickg-eval/log/publickg_roles_accuracy_20260907/`
contains `roles.json` and `final_states/{native,support,query}/index.json` plus
the indexed state files. Local aggregate results are in
`data/publickg_fixed_20260907/{quality,roles}.json`. Analysis worktree:
`/Users/philipp/projects/gfm/prodigy/.worktrees/role-topology`, branch
`codex/role-topology-interactions`, code HEAD `30df8cb5`. This file is private
and uncommitted. The one-page PDF remains unchanged.
