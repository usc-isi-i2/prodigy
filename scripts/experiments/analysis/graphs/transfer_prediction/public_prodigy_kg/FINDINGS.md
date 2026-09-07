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
This was the decision at the time of that entry. The support-only repair has
since completed in its separate worktree and did not beat U1; it is closed.

## Completed public pre-metagraph readout comparison (2026-09-07)

The saved-native comparison is complete: 500 episodes, 20 ways, 3 support and
4 query examples per class, 40,000 query occurrences. Same nominated public
checkpoint and exact previously captured episodes; native replay passed the
existing 1e-4 absolute, zero-relative tolerance. No new training or target tuning.
Runtime revision: `382cb1ea`, branch `codex/publickg-paired-state`.

| Deployment readout | Accuracy | Macro-F1 | OVR AUC | NLL |
|---|---:|---:|---:|---:|
| Native metagraph | .737950 | .721805 | .974829 | .873646 |
| Pre-metagraph ridge | .769050 | .744457 | .974552 | 2.823206 |

Ridge uses row-L2 embeddings, lambda=1, no intercept, scale=1, fitted only on
support labels separately per task. The stage is the actual input to the first
metagraph block after UX; it is not a social-graph pooling implementation.

Paired mean gains: accuracy +3.110 percentage points, macro-F1 +2.265 points.
Episode-bootstrap 95% intervals are [2.715, 3.500] and [1.821, 2.710] points,
respectively. Ridge wins accuracy in 355 episodes and loses in 98 (47 ties);
macro-F1 wins/losses are 334/166. AUC difference is -0.0278 points, interval
[-0.0915, +0.0351]. NLL worsens in every episode at the fixed scale.
Intervals are conditional on one checkpoint and one target; recurring entities
can induce episode dependence. They are not training-seed or domain intervals.

Interpretation: the pre-metagraph classification advantage extends to this
public knowledge-graph task, not only binary social graphs. This does NOT show
uniform metric superiority, harmful training gradients, or a new training
method. Native probabilities have much better NLL; the fixed ridge scale is
not calibrated, and probability quality must not be conflated with argmax
classification. AUC is essentially unchanged. Scalar temperature adjustment
cannot change either readout's argmax, so the accuracy gap is not solely a
scalar-temperature effect.

Authoritative Tucker artifacts (not mounted locally):
`/dataMeR1/phil/gfm/prodigy-publickg-paired-state/log/publickg_readout_20260907/`.
`execution_status.json` reports complete; `summary.json` contains all paired
metrics; 500 episode files preserve predictions, per-query losses, correctness
transitions, source hashes, and task/query mappings. Source protocol/index and
checkpoint hashes are recorded in `protocol.json`. The first launch failed an
import-isolation guard before creating results; its log is retained. The retry
matches the pinned upstream working directory and completed successfully.

Next high-information analysis: inspect corrected and corrupted query examples
and class-confusion changes. Test whether native inference introduces class-
specific decision biases versus changing within-class ranking. Do not fit a
new router or tune ridge on these inspected query labels.

### First example-level inspection

Across all 40,000 query occurrences, ridge corrects 3,363 native errors and
corrupts 2,119 native successes (net 1,244); both are correct on 27,399 and both
wrong on 7,119. Top-2 accuracy rises from .862100 to .887775. Similar OVR AUC
therefore does not imply preserved within-query class ranking.

Simple class collapse is not supported by the aggregate concentration check:
native mean maximum predicted class count is 7.558 versus ridge 7.730 (true
count is four per class); summed absolute deviation from four is 22.588 versus
24.704. These are descriptive checks, not a rejection of all class-bias models.

Earliest-in-stream examples (not selected by greatest gain), episode 00000:

- Query 8: Away From Her → New York Film Critics Circle award for best actress.
  Gold/ridge: award-winning work → award. Native: film → distribution medium.
- Query 10: Bodyguards and Assassins → Hong Kong Film Awards for Best Action
  Choreography. Same gold/ridge versus native confusion.
- Query 11: Midnight Velocity (dataset alias) → Chicago Film Critics Association
  award for best actor. Same confusion again.
- Query 43, counterexample: Cleopatra (1963 film) → Germany (dataset alias
  `Jermany`). Native correctly selects film runtime/release region; ridge
  incorrectly selects film distribution medium.

The actual relation paths are
`/award/award_winning_work/awards_won./award/award_honor/award` (class 2),
`/film/film/distributors./film/film_film_distributor_relationship/film_distribution_medium`
(class 8), and `/film/film/runtime./film/film_cut/film_release_region` (class 10).
Class mapping was verified by exact equality (maximum absolute error zero)
between saved input label embeddings and the released text-feature cache.
Entity pairs come from saved `center_node_idx` and released `entity2id.json`;
names are dataset aliases, not manually verified entity descriptions. The model
ignores these label embeddings in inference; they identify classes for auditing.

Working hypothesis: episode-specific relation confusions can override useful
head/tail type distinctions. This is NOT yet causal evidence: inspect the three
support pairs for competing classes and compare their pre/post-metagraph
geometry before attributing the confusion to support composition or inference.

### Support inspection and existing counterfactuals

The same episode's class-2 supports include Drugstore Cowboy → Independent
Spirit Award for Best Actor, an unnamed entity → MTV Movie Award for Best Song,
and an entity aliased as Elvin Tibideaux → Primetime Emmy guest-actor award.
Class-8 supports point to DVD (two cases) and streaming (one); class-10 supports
point to the United States (two) and Spain (one). Dataset aliases are noisy,
but tail categories distinguish these particular competing classes.

Existing saved support-context-removal interventions correct award queries
8, 10, and 11. Crossing only the changed final class-reference vectors into
native query vectors also corrects all three. Changing only query vectors
corrects query 11 but not 8 or 10. Thus for queries 8 and 10 the tested reference
substitution is sufficient to repair the decision with native queries fixed;
this is stronger localization than the semantic example inspection alone.

The counterexample matters: for query 43 (Cleopatra → Germany), native is
correct, removing support context is wrong, and reference-only replacement is
also wrong; query-only replacement preserves correctness. First-layer value-only
replacement does not fix queries 8 or 10. These facts do not support a universal
value-corruption story or global support-context removal (which already failed
badly in the full public aggregate). The effect is query/class dependent.

These are inspected examples within one episode, not independent replication of
a population mechanism. Next stage comparison should separate the utility of
post-metagraph example embeddings from the learned class references: ridge on
post-metagraph support/query embeddings versus the same ridge before the
metagraph, paired against native inference. This distinguishes representation
damage from a poor learned readout without transplanting vectors across stages.
