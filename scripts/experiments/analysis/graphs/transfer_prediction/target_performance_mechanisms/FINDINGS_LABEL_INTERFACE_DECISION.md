# Nominated test: label-initialization by support-context interaction

7 September 2026. Frozen decision before new outcomes. **COMPLETED** at code
`b377f3bc`; original nomination below is preserved.

## Completed result and decision

Tucker isolated checkout `/dataMeR1/phil/gfm/prodigy-label-context`, detached
`b377f3bc`; output `log/label_context_20260907`. DONE records 256 episodes,
two streams, eight cells. Five existing label-helper tests passed locally;
the new runner imports successfully. All 256 input/weight checks, exact
current-endpoint replay checks, exact table-row checks, within-interface
query-invariance checks, and flag-restoration checks passed during execution.

| Stream / initialization | Intact within AUC | Suppressed within AUC | Suppression change | Pooled AUC change | Accuracy intact / suppressed |
|---|---:|---:|---:|---:|---:|
| Original / current | .566406 | .654044 | +.087637 | +.059533 | .486979 / .251628 |
| Original / training table | .561921 | .565285 | +.003364 | -.026055 | .532227 / .458984 |
| Fresh / current | .560909 | .664424 | +.103516 | +.067289 | .496745 / .251302 |
| Fresh / training table | .550709 | .569372 | +.018663 | -.023736 | .532552 / .468750 |

The change in suppression effect under table initialization is -8.4274 and
-8.4852 within-episode AUC points. The predeclared positivity criterion is
still satisfied in both streams: do not relabel this as disappearance or
reversal of the primary effect. However, the large effect is clearly
initialization-dependent descriptively; a positivity-only claim would hide
the important magnitude change. Pooled effects reverse sign. No significance
claim or causal fraction is inferred from these conditional comparisons.

Suppressed global-positive prediction rates change from .99837/.99870 to
.60026/.59831. The near-all-positive collapse is not preserved with training
initialization, though suppression still lowers accuracy by 7.32/6.38 points.
Suppression remains harmful to accuracy. This factorial has no matched ridge
arm, so the separate-stream ridge result cannot establish a fair comparison
against this initialization change.

**Advisor update:** do not use the 50k evidence as strong proof of an
interface-independent learned transfer failure. A smaller within-episode
effect survives, but initialization strongly moderates its magnitude and
decision behavior. Current-interface K/V and prototype results remain valid
within their scope. The next explanation must incorporate label-reference
initialization rather than treating it as a negligible direct residual.
The between-interface difference can involve changed query processing and
routing, so this factorial alone does not identify which indirect path causes
the interaction. Do not expand into more label permutations or thresholds.

Private logits, audits, protocol and metrics are retained in the output above.
Only the runner code was pushed. The compatibility PDF predates this result
and must not be treated as the current final scientific argument.

The completed 540-cell interface study concerns older nine-source specialists
with intact supports, not the current HK50k suppression contrast. Direct label
residual accounting does not exclude indirect label effects on attention and
query processing. An independent contribution reviewer recommends this single
factorial before further component partitions or a broader mechanism claim.

## Feasibility verified read-only

### Paired follow-up: sensitivity depends on the quantity measured

Read both completed private prediction files without new forwards. Changing
initialization changes class decisions on 16.05%/15.36% of intact query
occurrences, versus 39.81%/40.04% under suppression. Yet its mean probability
total-variation distance is *smaller* under suppression: .02894/.02906 versus
.06801/.07332 intact. Thus it is incorrect to say suppression simply amplifies
the size of the output perturbation. Smaller probability movements change
many more decisions in the suppressed operating point, consistent with the
previously observed compressed contrast and near-boundary scores.

Training-table initialization corrects/loses 316/177 and 291/181 intact
decisions; under suppression it corrects/loses 930/293 and 949/281. These are
counts out of 3,072 query occurrences per stream, not independent accounts.

The within-episode AUC interaction (table suppression effect minus current
suppression effect) is negative/zero/positive in 98/7/23 original episodes
and 91/7/30 fresh episodes. Medians are -.04630 and -.02778, compared with
means -.08427 and -.08485. The mean interaction is not solely an artifact
of one exceptional episode; this is descriptive, not an independence-based
sign test or confirmation of a particular indirect path.

Prediction SHA256 values: original
`fb8147e1720a441688493baceadba574679fac557b3caaac318db17f4cd9081c`;
fresh `4dc83bb91e05b5fa915cbe776b8411a2c781f832fba29b6c2efe6876d454f6d4`.

### Initialization values inspected before the run

The nominated checkpoint retains `learned_label_embedding.weight` [1000,256].
Recorded NM training sets `ignore_label_embeddings=True`; the implementation
replaces projected labels by table rows indexed with `arange(label_count)`.
The table is frozen under the recorded configuration. Evaluation currently
sets the flag False. Restoring True recreates this label initialization, not
the full training task, episode cardinality, or training-mode protocol.

First saved original episode active label norms are 8.7989/9.3989 versus
16.8739/15.5363 for table rows 0/1; displacement norms are 19.2840/18.0005.
The active labels are identical for intact/suppressed support endpoints.
These are feasibility observations, not evidence of a performance effect.

## Fixed scope

- HK50k checkpoint already nominated, digest
  `c392cb264a05ac9a21dd6d3afe4590d3a9e7e8f74079da3e2a89fe35668a6a21`.
- Both existing political streams, all 128 episodes each, from
  `prodigy-classref50k/log/classref50k_20260907/private_inputs`.
- Four conditions: current initialization/intact, current/suppressed,
  training-table initialization/intact, training-table/suppressed.
- Change only `ignore_label_embeddings` for the initialization factor; no
  scaling, permutation, zeroing, calibration, new readout or checkpoint search.
- Use the existing audited suppression and flag-scoping helpers. CPU-only.
  No new training, sampling, or public result publication.

## Verification and outcomes

Before interpretation require original batch/weight digests, unchanged weights
and inputs, restoration of the flag, observed use of the exact first two table
rows, and reproduction of saved current-interface endpoints. Check query
invariance between intact/suppressed separately within each initialization.
Do not require or claim query invariance across initialization settings.
Save all logits, metadata, failures and a completion receipt privately.

Primary: within-episode AUC suppression effect for each initialization and
their difference, separately by stream. Also retain accuracy and global-class
prediction proportions. This is one checkpoint and reused streams, not seeds
or an unobserved-domain test. No outcome-driven threshold is selected.

If suppression improves under training-table initialization in both streams,
the mismatch is not necessary for this failure. If effects vanish/reverse,
the original result cannot be presented as interface-independent learned
negative transfer. Mixed signs are unresolved, not license to select a stream.
In either case, changed label initialization can affect queries and routing;
the between-initialization contrast is not value-only causal localization.

Historical nomination status: execution was pending. Code must move
to Tucker through Git under the existing private-results/public-code boundary.
Worktree `.worktrees/role-topology`, branch `codex/role-topology-interactions`,
HEAD `30df8cb5`. This decision record remains private and uncommitted.
