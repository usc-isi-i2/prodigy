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

### Completed before/after readout comparison

Runtime `c0f74e18`, same 500 saved public episodes and checkpoint, completed at
`log/publickg_stage_readout_20260907` on the dedicated Tucker public worktree.
Native replay passed the existing tolerance. No new target tuning.

| Readout | Accuracy | Macro-F1 | OVR AUC |
|---|---:|---:|---:|
| Native | .737950 | .721805 | .974829 |
| Ridge before metagraph | .769050 | .744457 | .974552 |
| Ridge after metagraph | .701350 | .671331 | .964031 |

Post minus pre: -6.770 accuracy points and -7.313 macro-F1 points; post loses
accuracy in 460 episodes, wins 22, ties 18. Thus replacing final class references
with an ordinary support-fitted ridge is not a general repair. It does not
follow that all information is destroyed: support representations are label-
conditioned whereas queries are not, so cross-role alignment is a live alternative.

Saved `geometry` contains actual pre-metagraph embeddings, actual decoder-input
embeddings, and final class references for all examples, allowing role-shift
analysis without another model run. A first descriptive pass finds strongly
negative average cross-role cosine after inference, unlike the positive cosine
before inference. Class-conditioned separation estimates from that pass assumed
class-block ordering and are not promoted to evidence until the support-label
mapping is verified. The class-free role shift and the completed readout scores
motivate a bounded test of separate support/query centering, not a new claim of
novelty or a declaration that all post-metagraph geometry is worse.

### Completed fixed centering diagnostic

Runtime `a918531b`; all 500 episodes complete in Tucker
`log/publickg_role_centering_20260907`. The script reads actual native support
labels, verifies native artifact hashes, and reproduces both uncentered ridge
controls within 1e-5. Row-L2 normalization follows centering; lambda=1, no
intercept, scale=1 throughout. No labels from queries enter the fitted readout.

| Stage / centering | Accuracy | Macro-F1 | OVR AUC |
|---|---:|---:|---:|
| Native inference | .737950 | .721805 | .974829 |
| Pre / none | .769050 | .744457 | .974552 |
| Pre / support mean for both roles | .803475 | .787986 | .977813 |
| Pre / separate role means | .807525 | .793128 | .977929 |
| Post / none | .701350 | .671331 | .964031 |
| Post / support mean for both roles | .319900 | .275742 | .957936 |
| Post / separate role means | .721600 | .696859 | .958213 |

The purely support-fitted pre-metagraph centered baseline exceeds native by
6.5525 accuracy points and 6.6181 macro-F1 points. Accuracy wins/losses/ties are
448/27/25 episodes. Its NLL is 2.673942 versus native .873646 at these fixed
scales: not uniform probabilistic superiority. Separate means add only .405
accuracy points before inference and use unlabeled query distribution.

Post separate centering recovers 2.025 accuracy points, but remains 1.635 below
native and 8.1875 below support-centered pre-metagraph ridge. A simple global
role offset is insufficient to explain the loss of support-readout utility.
Support-based centering helps before inference but catastrophically fails
afterward, consistent with different support/query geometry. It does not alone
prove the cause is role asymmetry rather than other learned transformations.

This diagnostic was designed after inspecting this same stream. Treat gains as
exploratory until replicated with a frozen rule on a new stream/checkpoint.
Do not market centering as novel: SimpleShot already studies mean subtraction
and L2 normalization (https://arxiv.org/abs/1911.04623), and Prototype
Rectification studies feature shifting for support/query bias
(https://arxiv.org/abs/1911.10713). The prospective contribution is explaining
and avoiding inference-induced incompatibility of reusable representations,
not renaming these existing readout ingredients.

Advisor decision: retain support-centered pre-metagraph ridge as the stronger
deployment baseline; stop trying global centering variants to rescue the final
stage. Next, freeze this rule and test breadth/replication before more elaborate
architectural proposals. A method must beat this baseline, not only native or
TRACE, to earn a deployment contribution.

### Frozen fresh-stream replication completed

Runtime `628fec69`; Tucker output `log/publickg_readout_fresh_20260907` reports
complete. Fixed rule nominated before this run: subtract the support mean from
both roles, row-L2 normalize, ridge lambda=1, no intercept, scale=1. Test sampler
seed explicitly changed from448 to100451; global seed0, checkpoint, data pools,
500 episodes and parity tolerance remain fixed. All 500 saved episode hashes
were checked after completion. First three episode entity-pair arrays and label
embedding arrays differ from the corresponding discovery episodes. This verifies
the sampler change, not disjoint entities or independent training.

| Frozen readout | Accuracy | Macro-F1 | OVR AUC | NLL |
|---|---:|---:|---:|---:|
| Native | .735350 | .719447 | .975168 | .873143 |
| Uncentered pre-metagraph ridge | .768125 | .744370 | .974447 | 2.823322 |
| Support-centered pre-metagraph ridge | .802800 | .787727 | .978340 | 2.674026 |

Centered-minus-native: +6.745 accuracy points, +6.828 macro-F1 points,
+0.3172 OVR AUC points. Accuracy wins/losses/ties:463/19/18. Uncentered ridge
wins/losses/ties:365/89/46. Probability loss remains worse at the frozen scale.
No new algorithm or independent training-seed/domain replication is claimed.
The discovery gain survives a genuinely changed episode stream without tuning.

Next causal distinction: cross-fit support embeddings in the query role, hiding
each held-out support label during its representation pass. This addresses the
strong alternative that self-label-conditioned support embeddings are unsuitable
training examples for a query readout. Use discovery episodes for this mechanism
test, not the just-completed frozen replication stream. No cross-fit outcome is
available yet.

### Query-role cross-fit pilot completed: partial recovery, no deployment advance

Runtime `914a1235`; all first64 discovery episodes complete at Tucker
`log/publickg_crossfit_20260907`. Three passes hide one support per class,
zero its label-bearing edge value and label row, set its query role, and clear
unused sequence inputs. Only pinned non-sequence MetaGNN is allowed. Each
pass restores captured native buffers/modes/RNG. Held support embeddings replace
their native versions; original query embeddings remain unchanged. Native
logit parity checked before every episode. Actual source hashes retained.

| Readout on the same64 episodes | Accuracy | Macro-F1 | OVR AUC |
|---|---:|---:|---:|
| Native | .751758 | .735109 | .977927 |
| Pre ridge | .779883 | .754228 | .976958 |
| Pre support-centered ridge | .815625 | .799545 | .980867 |
| Post ridge | .712695 | .682049 | .967630 |
| Post support-centered ridge | .341797 | .296781 | .962839 |
| Cross-fit post ridge | .741211 | .715731 | .967231 |
| Cross-fit post support-centered ridge | .742578 | .718252 | .966802 |

Cross-fit post gains2.8516 accuracy points versus ordinary post ridge but remains
1.0547 points below native and7.4414 below centered pre ridge. AUC does not
recover. The catastrophic post-centering result is largely relieved by role-
matched support encoding, consistent with a role mismatch contribution. The
test changes support conditioning from three to two labeled examples per class,
so do not interpret it as a clean additive mediation estimate. It is a first64
discovery pilot, not independent evidence of a general causal law.

Advisor decision: this is not a competitive cross-fit deployment method. Do not
expand it to the frozen fresh stream or search more centering variants. Retain
the bounded mechanistic conclusion: role mismatch contributes, but tested role
matching and simple offsets do not recover pre-metagraph readout utility. The
next paper-threatening alternative is that generic/random encoder features plus
the stronger readout explain the gain, rather than useful pretraining. Test that
before proposing another training architecture.

### Saved-initialization / endpoint-text controls completed

Runtime `01104df2`, 500 discovery episodes, completed Tucker output
`log/publickg_pretraining_control_20260907`. Exact saved initialization from the
same public training run, not a newly chosen random seed. Its own BN buffers
are preserved; saved episode modes/RNG are restored, and extraction stops before
the metagraph. Endpoint-text features concatenate actual head/tail 768d vectors;
pooling-edge order is validated. All arms use the same ridge settings.

| Features / centering | Accuracy | Macro-F1 | OVR AUC |
|---|---:|---:|---:|
| Trained / none | .769050 | .744457 | .974552 |
| Trained / support | .803475 | .787986 | .977813 |
| Initialization / none | .720000 | .694047 | .961539 |
| Initialization / support | .700750 | .677272 | .955100 |
| Endpoint text / none | .719850 | .701265 | .958658 |
| Endpoint text / support | .721250 | .702946 | .958032 |

Thus the centered trained encoder beats the better tested initialization rule
by8.3475 accuracy points and the better tested endpoint-text rule by8.2225.
Even without centering, trained features exceed initialization by4.905 points.
Native's gain over centered endpoint text is only1.67 points. This supports
useful GFM pretraining whose classification benefit is underexposed by native
inference. It does not establish that pretraining always helps, and every arm
uses pretrained text features. The control is one saved initialization/target.

### Advisor's cross-target selection proposal tested, not supported

Read-only pilot on the existing matched-schedule audit: for each of four targets,
each rung2/3/4 and training seed0/1/2, select one of blocked/interleaved/replay100
using mean original-stream AUC on the other three targets. Compare selection by
full versus uncentered U1 AUC; evaluate BOTH selected models with U1 on the held-
out target's fresh stream. Lexicographic schedule order breaks exact ties.
This is retrospective and uses the existing uncentered readout, not the new rule.

Among36 decisions,13 selections change. U1-based selection wins4, loses9, ties23;
held-out mean U1 AUC is .838245 versus .842369 for full-based selection
(delta -0.41235 points). Mean deltas are nonpositive on all four targets.
Source: `/dataMeR1/phil/gfm/prodigy-schedule-stage-audit/log/stage_audit_20260907/cells.json`,
SHA256 `39d3f563399fc3cfe517d6a3be16342aebe6ec919c01bce9976a2f42177fee54`.

Decision: do not claim a successful cross-target stage-aware selector. Ranking
reversals alone do not imply that validation on other domains predicts the best
representation on a new target. The public readout/pretraining results remain
supported; a model-selection framework is not established by this pilot.

### Frozen rule applied to social models: completed breadth test

Runtime `0a223848`, completed Tucker output
`log/social_centered_readout_20260907`. Reused saved embeddings from all eight
rung4 seed0 native/joint/isolated/ridge-only × blocked/interleaved models,
five targets, two streams:80 prediction-export receipts and400 readout cells.
Cached inputs were rehashed and uncentered ridge controls reproduced before
scoring. Macro-F1/accuracy use production global-probability tie handling.

Primary comparison: ordinary native-trained models, fresh fixed-four-target means.

| Schedule / readout | Accuracy | Macro-F1 | AUC | NLL |
|---|---:|---:|---:|---:|
| Blocked / full | .786947 | .780686 | .830391 | .625001 |
| Blocked / U1 | .802327 | .795349 | .846822 | .591297 |
| Blocked / centered U1 | .808757 | .802451 | .860847 | .517886 |
| Interleaved / full | .765788 | .759440 | .816190 | .780472 |
| Interleaved / U1 | .803141 | .797719 | .840030 | .568712 |
| Interleaved / centered U1 | .806559 | .800788 | .844000 | .520637 |

Centered raw-feature means are .731364 accuracy/.721158 macro-F1/.790131 AUC,
below both centered native-trained U1 means. Centered U1 minus full is positive
for accuracy/F1/AUC and negative for NLL in both streams, both schedules, and
both fixed4/all5 panels. Fresh fixed4 macro-F1 gains are2.1765 and4.1348 points;
AUC gains3.0456 and2.7810 points. Unlike public20-way results, social NLL improves.

Centering alone is not uniformly helpful over U1: on fresh twibot20 it improves
blocked accuracy2.4089 points but reduces interleaved accuracy.5534 points;
interleaved covid_political also declines. Do not describe per-cell dominance.
Breadth is retrospective across existing social outcomes with a public-fixed
rule, not new independent training seeds. The eight objective arms do not create
eight independent validations of the native-training claim.

Advisor synthesis: the fixed intermediate readout now has public fresh-stream
replication, an exact-initialization control showing useful pretraining, and
positive social panel breadth. This is a stronger baseline contribution than
TRACE, but neither universal superiority nor a successful cross-target schedule-
selection algorithm has been established.

### Fresh-stream initialization and text controls: complete

Runtime `8031c57c`, Tucker `log/publickg_pretraining_control_fresh_20260907`,
500/500 episodes and `execution_status.json: complete`. Source is
`publickg_readout_fresh_20260907`; index SHA256
`60f9317ec226225b4e5a33c47ecc007dac2484573936ef851e087ab5c2778094`.
Initialization SHA256 `83ed396209297c880097f96fe84f25f00ed90984ed0cc058670d42cb63bfc4f8`.
Each source artifact was rehashed before reading embedded paired geometry.

| Readout | Accuracy | Macro-F1 | OVR AUC | NLL |
|---|---:|---:|---:|---:|
| Native | .735350 | .719447 | .975168 | .873143 |
| Trained / uncentered | .768150 | .744393 | .974447 | 2.823322 |
| Trained / support-centered | .802800 | .787727 | .978340 | 2.674026 |
| Initialization / uncentered | .720700 | .695678 | .961288 | 2.771032 |
| Initialization / support-centered | .697775 | .674303 | .955725 | 2.717717 |
| Endpoint text / uncentered | .714750 | .697192 | .957863 | 2.774713 |
| Endpoint text / support-centered | .716100 | .698477 | .957220 | 2.761918 |

Centered pretrained accuracy exceeds the stronger tested initialization control
by 8.21 points and the stronger text control by 8.67 points. Native exceeds that
text control by only 1.925 points. Centered pretrained macro-F1 exceeds the best
initialization by 9.20485 points and best text by 8.92498 points. This supports
useful GFM pretraining on the fresh episode stream, not another training seed.
Public NLL remains worse than native, so the conclusion concerns classification
and ranking utility, not calibration dominance.

The recomputed uncentered readout accuracy differs by .000025 (one of 40000
queries) from the online saved score .768125 despite the required 1e-5 logit
parity check passing. Do not silently replace the original online result or call
this bit-exact prediction replication. Centered and native aggregate results
match the original fresh-stream report. The tiny uncentered discrepancy needs
an example-level tie/numerical check before combining those rows in a final table.

### Cardinality discriminator: the gap persists at training-matched 15-way

Runtime `7bd7047a`; outputs `publickg_cardinality_15way_20260907` and
`publickg_cardinality_20way_20260907` under the paired-state Tucker log root.
Both completed 128 episodes. All 256 artifact hashes and ordinal sequences
verified before aggregation. Fixed seed0 checkpoint step8000, 3shot/4query,
200 candidate relations, sampler seed300455; centered rule unchanged.

| Ways | Native accuracy | Centered accuracy | Native macro-F1 | Centered macro-F1 |
|---|---:|---:|---:|---:|
| 15 | .780208 | .834115 | .767356 | .822119 |
| 20 | .743262 | .808301 | .727486 | .792750 |

Accuracy gains are 5.390625 and 6.503906 points; macro-F1 gains 5.476352 and
6.526418 points. Accuracy wins/losses/ties:105/13/10 and113/7/8 respectively.
Native/centered OVR AUC:15-way .976186/.979762;20-way .974598/.979168.
Native/centered NLL:15-way .746230/2.360683;20-way .876431/2.670160.

The advantage survives matching training's 15-way cardinality: the 15-to-20
mismatch is not sufficient to explain the observed gap. The ~1.11-point larger
accuracy gap at20-way is descriptive, not a proven interaction. Cardinality
also changes total examples and native BN context; examples across the two
arms are not paired. This does not by itself prove domain-shift causation.

Index SHA256:15-way `7dc942dddf1d587f075237cc0b38247ea5e90d65bdd6053f320ace53abb05b9e`;
20-way `210e1e842a5d2ee2962416416cf3e1bde32c6f54990d52a4ce2a343a35ea9836`.

### Native-win audit before proposing a router

Read all native-trained cells in `social_centered_readout_20260907/summary.json`
across five targets, two schedules, and both streams. No target/schedule has a
positive native-minus-centered accuracy or macro-F1 contrast in both streams.
The original blocked covid_political macro-F1 advantage is .2462 points but
becomes -.2588 on fresh; blocked suspended .8000 becomes -.1987. These original
wins also disappear against the better of centered and uncentered U1.
Election2020 contains exact/near ties, not a substantial positive regime.
This does not exclude predictable episode-level wins, but a stable task-level
crossover is not established. Do not build or advertise a router on this evidence.

Numerical audit closure: the one-query fresh uncentered recomputation discrepancy
is episode401/query25, true class6. Online top scores: class2 .1757591665,
class6 .1757584661; recomputed class6 .1757591963, class2 .1757589728.
Maximum absolute logit difference is7.30157e-7, within the recorded tolerance.
This is a near-tie flip, not different examples or an aggregate arithmetic error.
Retain the original online metric when reporting that run.

### Cora breadth: deployment improvement without positive representation transfer

Completed runtime `f5db493d`, output `log/cora_readout_breadth_retry1_20260907`.
Both strict checkpoint loads and all batch trace-parity checks passed.128
seven-way/3shot/4query episodes per model;32 identical cached batches.
All32 input and64 output file hashes verified;256 model-episode metric rows.
Custom stratified test pool and transductive graph, not standard Cora benchmark.
Initial failed loader attempt retained separately; no predictions preceded fix.

| Checkpoint/readout | Accuracy | Macro-F1 | OVR AUC | NLL |
|---|---:|---:|---:|---:|
| Blocked/native | .407924 | .383682 | .773740 | 1.588211 |
| Blocked/U1 uncentered | .402623 | .348499 | .763428 | 1.913780 |
| Blocked/U1 centered | .438895 | .403355 | .766276 | 1.811580 |
| Interleaved/native | .311384 | .292621 | .681106 | 1.818053 |
| Interleaved/U1 uncentered | .322545 | .275823 | .689790 | 1.916137 |
| Interleaved/U1 centered | .386719 | .352346 | .734956 | 1.841651 |
| Raw uncentered (same both) | .573661 | .547938 | .849726 | 1.867020 |
| Raw centered (same both) | .571150 | .547006 | .848331 | 1.822570 |

Centered U1 gains3.0971/7.5335 accuracy points over native for blocked/interleaved,
but raw uncentered exceeds centered U1 by13.4766/18.6942 points. Thus this task
broadens the deployment-bypass observation, NOT the claim that useful GFM gains
are hidden. Representation utility relative to text is poor here. Without a
same-architecture initialization control, do not attribute that deficit solely
to training rather than architecture. Blocked native also retains higher AUC
than centered U1. This boundary must remain in the paper; do not search for a
replacement favorable target or describe all metrics as improving.

### Fixed-first-batch Cora example inspection

Inspected the first five blocked-model queries in batch000 where raw uncentered
is correct and centered U1 is wrong (query6,7,13,29,35). Mapped each episode's
local class columns to graph labels using actual center IDs. Context counts
use saved sampled real nodes, excluding the center and artificial supernodes;
labels are used only for this retrospective diagnosis, never prediction.

Query13/node2403, *Reasoning with Portions of Precedents*, is Case Based;
raw is correct, while U1 and native predict Genetic Algorithms. All seven
sampled context nodes are Case Based. Query35/node573, *Iterated Revision and
Minimal Change of Conditional Beliefs*, is Probabilistic Methods; U1/native
predict Theory despite all nine sampled context nodes being Probabilistic
Methods. Query29/node544 is labeled Neural Networks; U1/native predict
Probabilistic Methods despite12 of15 sampled context nodes being Neural Networks.
Its statistical-decision-theory abstract also illustrates that dataset labels
and textual topic intuition need not align cleanly.

These examples contradict a universal wrong-neighborhood-majority explanation,
not every form of harmful aggregation. They do not distinguish learned encoder
distortion from support-set geometry and cannot establish population causation.
No intervention or hyperparameter choice was selected from these examples.

### Context sensitivity: factual associations help both targets

Completed `context_sensitivity_retry1_20260907`, runtime6022c435. The initial
attempt completed Cora but failed on second-target logger configuration; retained
separately, not used for effect selection. The retry completed both32-batch arms,
128episodes each, with256 metric rows and64 output hashes verified. Strict
checkpoint loading and factual saved-U1/native parity passed for every batch.
Within-episode permutation preserved center/synthetic features and non-feature
inputs; shuffled real background features only. Same blocked2500 checkpoint.

| Target | Factual centered-U1 accuracy | Shuffled accuracy | Change (points) |
|---|---:|---:|---:|
| Cora | .438895 | .193359 | -24.5536 |
| covid_political | .864583 | .607096 | -25.7487 |

Macro-F1 changes: -22.4066/-26.3443 points respectively. OVR AUC changes:
-19.5905/-23.8571 points; NLL increases .115450/.164034. Raw centered accuracy
is .571150/.754557, respectively. These are episodewise metrics; social macro-F1
and AUC should not be equated with earlier pooled-global panel summaries.

Decision: reject the nominated hypothesis of context disruption helping Cora
while harming the positive-transfer target. Factual feature-neighborhood
associations are useful relative to shuffled ones on both, even though Cora's
learned representation remains below text. This does not prove factual graph
processing beats a no-context encoder; shuffling induces unnatural context.
Do not promote context scrambling as a repair or claim a demonstrated
target-adaptive context rule. The observed Cora representation deficit is not
explained by this proposed intervention, and the opposite-sign branch is closed.

## Exploratory support-validation safeguard (2026-09-07)

Read-only CPU diagnostic on the completed `cora_readout_breadth_retry1_20260907`
saved batches and embeddings. For each episode, leave out each support example,
refit support-centered ridge on the remaining supports, and select raw versus
U1 by mean held-out support accuracy (ties select raw). Query labels enter only
the final accuracy evaluation. All folds retain every class. No tuning performed.

| Model | Fixed raw accuracy | Fixed U1 accuracy | Selected accuracy | Raw selected |
|---|---:|---:|---:|---:|
| Blocked | .571150 | .438895 | .562779 | 118/128 |
| Interleaved | .571150 | .386719 | .565290 | 122/128 |

Mean support-CV accuracy: raw .511905; U1 blocked .356027, interleaved .291295.
This retrospective check recovers much of the fixed-text advantage but does not
beat fixed text. It is a deployment baseline, not a new contribution or an
explanation of representation damage. Only accuracy was computed in this quick
diagnostic; do not imply corresponding F1/AUC gains. Results are console-derived,
not yet a packaged, receipt-verified experiment artifact.

## Next nominated test: hidden returns to additional pretraining

Before inspecting intermediate checkpoint target results, nominate saved public
seed-zero checkpoints `state_dict_2000`, `state_dict_4000`, `state_dict_8000`
(2001, 4001, 8001 optimizer updates under the native loop). All three files were
verified present under the native training output's `state/` directory.
Use the first 128 episodes of the existing fresh 500-episode stream identically
for each checkpoint; retain the fixed support-centered U1 rule and native
normalization protocol. No checkpoint selection or readout tuning. Compare the
2001-to-8001 and 4001-to-8001 changes, not only endpoint gaps. The already observed
8001-update endpoint means this is a discovery test, not a wholly unseen panel.

Prediction: additional training improves U1 utility after native performance
stalls or declines. If both improve similarly with a constant gap, do not claim
hidden training returns; if both saturate, close this direction. Report accuracy,
macro-F1, AUC and NLL separately. A positive seed-zero pattern must be tested at
the same nominated checkpoints on seed one; do not select its checkpoints based
on target outcomes. Seed one is still running and supplies no target result yet.

Novelty guardrail (primary-source abstracts checked 2026-09-07):
[Kumar et al., ICLR 2022](https://arxiv.org/abs/2202.10054) already demonstrate
linear probing outperforming fine-tuning under shift and explain feature
distortion from parameter updates. Our frozen inference comparison does not
perform target-side parameter updates, so their mechanism cannot simply be
claimed as ours. [Yang et al.](https://arxiv.org/abs/2404.01204) already study
downstream capability trajectories across language-model pretraining checkpoints.
Neither a probe advantage nor plotting intermediate checkpoints is novel alone.
The nominated experiment would matter if native and intermediate evaluations
give different conclusions about the returns to the *same* extra pretraining,
and that divergence replicates. This is a candidate distinction, not an
established novelty claim or a substitute for a fuller related-work comparison.

### Trajectory outcome: nominated hidden-improvement prediction not supported

Completed `log/publickg_trajectory_20260907` on Tucker paired-state at revision
`40f56ccb`. Three fixed checkpoints x128 identical fresh-stream episodes. All384
output hashes verified, and every8k native/centered-logit replay passed absolute
1e-4 parity (rtol0). Each checkpoint retained its own buffers; captured inputs,
RNG and native train modes were shared. Eight local tests passed before launch.

| Nominal checkpoint | Native accuracy | Centered U1 accuracy | Native macro-F1 | U1 macro-F1 | Native AUC | U1 AUC |
|---|---:|---:|---:|---:|---:|---:|
| 2000 | .774023 | .818945 | .760251 | .805118 | .982320 | .982342 |
| 4000 | .764941 | .813867 | .750337 | .799024 | .981483 | .981763 |
| 8000 | .750391 | .818066 | .735312 | .804048 | .976462 | .980212 |

Native NLL .715275/.732774/.846310; centered U1 NLL
2.689355/2.674437/2.670065. The logits retain fixed scale1; probability quality
is not equivalent to classification utility.

2000-to8000 accuracy changes: native -2.363281 points, U1 -0.087891 points.
4000-to8000: native -1.455078 points, U1 +0.419922 points. Do not cherry-pick
the latter small recovery to claim additional training improves representations.
The stronger nominated hidden-returns prediction is not supported: intermediate
classification utility is approximately flat across the full interval rather
than improving. Native predictions worsen more than the fixed intermediate
readout. This is a candidate *divergence in deterioration*, not evidence of
hidden scaling gains or destroyed information. The native/U1 accuracy gap grows
from4.492188 to6.767578 points. Replication is needed before elevating even this
narrower trajectory pattern. One seed, one target, recurring query entities;
128 episodes do not constitute128 independent training replications.

Decision: retain the trajectory as evidence limiting the paper's interpretation;
do not add a claim that longer pretraining improves representations. Do not
expand checkpoint search after seeing these outcomes. The fixed second-seed
replication remains the next validation step, not another repair sweep.

### Exact query-transition audit of the trajectory

Read-only CPU comparison of all128 paired episodes (10,240 query occurrences),
after re-verifying all384 output hashes, identical source receipts, and exact
query-label order across checkpoints. Counts compare2000 to8000, not independent
accounts or training replications:

| Rule | Correct both | Wrong both | Corrected | Corrupted |
|---|---:|---:|---:|---:|
| Native | 6783 | 1413 | 901 | 1143 |
| Centered U1 | 7725 | 1202 | 652 | 661 |

The nearly flat U1 aggregate masks1313 correctness changes. Do not describe it
as unchanged representations or invariant example-level behavior. Of1143 native
regressions,587 have U1 correct at both checkpoints;329 also regress under U1.
Of901 native recoveries,494 have U1 correct at both checkpoints. Thus a subset
isolates a deployment-level discrepancy, but the aggregate does not causally
localize drift to metagraph weights; encoder and solver are co-trained.

First three lexicographically selected native-regressed/U1-correct-both examples:
episode0 query28, local truth7, native7->4; episode0 query52, local truth13,
native13->16; episode0 query63, local truth15, native15->17. U1 predicts the
truth at both checkpoints in each. Local labels are episode-specific class
indices, not semantic relation names. These are inspectable saved example
identifiers, not a semantic explanation or a hand-picked success gallery.

## Nominated label-code assignment check

Code inspection confirms semantic label embeddings are ignored by the public
recipe; a semantic-label-prior explanation is therefore inapplicable. The model
instead indexes a fixed random embedding table by class position. The next
bounded check uses the fixed8k checkpoint and first32 fresh episodes, reversing
only table rows0..19. Inputs, support labels, graph edges and output columns
remain unchanged. No search over assignments. Compare identity/reversal query
disagreement, accuracy, macro-F1, AUC, NLL and logit changes. Require factual
parity and unchanged U1 embeddings before interpretation.

This tests label-code assignment invariance, not ordinary graph-node permutation
equivariance (which would move features and edges together). Material dependence
would identify an arbitrary nuisance input to native inference; improvement from
one reversal alone would not establish systematic harm or a useful repair.
Negligible dependence closes this branch. The fixed128-episode trajectory's
prediction counts show no collapsed class position: native8k counts range475..569
against512 true query occurrences per slot. This does not prove assignment
invariance. U1 has position-dependent accuracy too, so slot accuracy alone is
not causal evidence of anchor effects.

Runtime audit: the native initial and8k `learned_label_embedding.weight` tensors
are bit-identical (shape1000x256, max difference0). The first assignment run,
`publickg_anchor_assignment_20260907` at `e603a845`, failed its bit-exact U1
check on episode0 and is retained as failed. Repeated unchanged forwards on
that episode showed U1 drift up to7.6294e-6 and native-logit drift up to7.6294e-6,
with zero prediction disagreements. Reversal U1 drift was4.7684e-6; thus exact
equality of separately recomputed encoder outputs is not a valid runtime
assumption on this GPU. No aggregate intervention result was accepted.

Revised isolation protocol, before aggregate outcomes: for the reversed arm,
validate natural pre-M data embeddings against factual U1 at existing public
atol1e-4/rtol0, then explicitly replace those data rows with cached factual U1.
Leave reversed label rows untouched. Require bit-exact captured U1 after this
clamp and full model-state restoration. This tests changing label codes with
representations literally fixed, rather than accepting uncontrolled numerical
drift. Keep first32 episodes and the single reversal unchanged.

### Assignment outcome: modest decision sensitivity, not a repair result

`publickg_anchor_assignment_retry1_20260907` completed at `b8e939e8`, all32
output hashes verified. All32 U1 clamps were bit-exact; factual native replay
max error9.5367e-6 and natural reversed-U1 drift max9.5367e-6. Every model-state
restoration check passed. Six unit tests passed before deployment.

| Assignment | Accuracy | Macro-F1 | OVR AUC | NLL |
|---|---:|---:|---:|---:|
| Identity | .742578 | .725439 | .976295 | .868602 |
| Reverse | .744141 | .726645 | .976434 | .862749 |

86/2560 query predictions differ (3.359375%); maximum absolute logit difference
is1.648638. Changes in accuracy/macro-F1 are only +0.15625/+0.12052 points.
Thus arbitrary label-code assignment can alter individual native decisions with
representation features fixed. This single reversal does not establish a
systematic performance penalty, explain the intermediate-readout advantage, or
provide a beneficial deployment rule. No assignment search, second permutation,
or claim of a substantial average repair follows from these data. Keep this as
a limited sensitivity finding rather than the central contribution.

## Nominated temporal component crossover

Independent review identifies one unresolved alternative to a probe/head
mismatch: whether the observed2k-to8k native regression follows changes in the
inference component, rather than changes in encoder information or compatibility.
Fix a2x2 crossover of encoder checkpoint2000/8000 and inference checkpoint2000/8000
on the same128 trajectory episodes. No target-fit alignment or additional
checkpoint search. Both diagonal outputs must reproduce the saved trajectory.

Boundary from pinned S2,UX,M2 implementation: encoder owns `layer_list.0`,
`layer_list.1`, and `initial_input_mlp`; inference owns `layer_list.2`,
`initial_label_mlp`, `learned_label_embedding`, both final MLPs and `logit_scale`.
Assign normalization buffers with their containing module. Reject unclassified
state keys. Keep each encoder's saved U1 literally fixed across inference
conditions, after checking naturally computed U1 at public numerical tolerance.
The initial-label path is ignored in this recipe and final MLPs are identities,
but their ownership remains explicit rather than silently inherited.

Prediction before crossing outcomes: early inference restores late-encoder
performance and late inference worsens early-encoder performance. This would
localize a conditional effect of inference changes during training, not explain
the entire U1 advantage. If both crossings fail, incompatible co-adapted
components prevent attribution; close without alignment searches. Compare
accuracy/macro-F1 as well as AUC/NLL, do not call probability-only changes a
classification rescue. This is a discovery test; second-seed replication remains
necessary for a general temporal claim.

### Crossover outcome: encoder changes hurt native readout; inference changes help

`publickg_crossover_20260907` completed at `99524810`. All512 output hashes
verified; both128-episode diagonal conditions reproduce trajectory predictions
at1e-4 absolute tolerance; each encoder's U1 is held bit-exact across inference
conditions. Assembled model-state restoration passed. No alignment or fitting.

| Encoder | Inference | Accuracy | Macro-F1 | OVR AUC | NLL |
|---|---|---:|---:|---:|---:|
| 2000 | 2000 | .774023 | .760251 | .982320 | .715275 |
| 2000 | 8000 | .780957 | .768567 | .982486 | .703857 |
| 8000 | 2000 | .731055 | .714448 | .973400 | .898459 |
| 8000 | 8000 | .750391 | .735312 | .976462 | .846310 |

The nominated early-inference-rescue prediction is contradicted, not merely
inconclusive. Late inference improves both encoders: +0.693359 accuracy points
with the early encoder and +1.933594 with the late encoder. Late encoder features
reduce accuracy under both inference modules: -4.296875 points under early
inference and -3.056641 under late inference. Macro-F1, AUC and NLL agree in
direction. Meanwhile the fixed support-fitted U1 readout is .818945 early versus
.818066 late. The best native combination is early encoder plus late inference,
not either complete checkpoint; this is an observed test-set comparison, not a
validated checkpoint-selection/deployment algorithm.

Interpretation: the native training-time regression cannot be attributed to the
inference component simply deteriorating. Conditional crossover effects locate
the adverse change in the encoder outputs *as consumed by these native inference
modules*, while inference changes partly compensate. An approximately preserved
support-fitted readout does not guarantee preserved utility for a learned
inference rule. Conversely, worse native predictions do not demonstrate erased
classification information. This is readout-dependent transfer deterioration,
not a proof of information invariance or a complete geometric explanation.

Scope: one seed, one target, two prespecified checkpoints; public native
training-mode BN is preserved, so normalization-mediated effects remain possible.
Do not claim an architecture-independent mechanism. The strong original
prediction was rejected, and the observed opposite pattern needs replication
before elevating it to the paper's central explanation.

## Nominated normalization sensitivity check

Pinned upstream `trainer.py` confirms a meaningful protocol difference:
standalone `eval_only` exits after an initial `do_eval` whose `model.eval()`
call is commented out, while periodic validation during training explicitly
calls `model.eval()`. Both encoder and metagraph use standard BatchNorm1d with
running statistics. Public results faithfully preserve the standalone path,
but this does not rule out a normalization-mediated explanation.

Fix checkpoint2000/8000 and first128 fresh episodes. Compare the existing
batch-statistics protocol with ONLY BatchNorm modules switched to frozen running
statistics. Use each checkpoint's own buffers, shared inputs/RNG and unchanged
remaining module modes. No recalibration, fitting, or running-stat transfer.
Evaluate native and centered U1 readouts under each condition; unlike the
crossover, U1 may change when encoder normalization changes and must not be
clamped to the original condition. Require original-condition trajectory parity
and full state restoration. Record both outcomes regardless of which is better.
This tests dependence on a concrete released evaluation choice, not a new
deployment method or a reason to silently replace the original protocol.

Read-only checkpoint audit confirms all four BatchNorm buffers have trained
statistics: `num_batches_tracked` is2021 at2k and8021 at8k; every running variance
is finite and strictly positive. Therefore the frozen-statistics comparison
does not substitute uninitialized buffers. It still tests source-estimated
statistics under target shift, not a universally preferred normalization rule.

### Normalization outcome: gap persists with checkpoint statistics

`publickg_normalization_20260907` completed at `98ba23d1`. All256 output hashes
verified, representing2 checkpoints x128 episodes x2 normalization conditions.
Original batch-statistics native/U1/ridge parity passed; all model-state and
mode-restoration checks passed. Seven tests passed before deployment.

| Checkpoint | Statistics | Native accuracy | U1 accuracy | Native macro-F1 | U1 macro-F1 |
|---|---|---:|---:|---:|---:|
| 2000 | Batch | .774023 | .818945 | .760251 | .805118 |
| 8000 | Batch | .750391 | .818066 | .735312 | .804048 |
| 2000 | Running | .756543 | .797461 | .742332 | .782454 |
| 8000 | Running | .701953 | .808301 | .682405 | .792527 |

Running-statistics native AUC/NLL:2k .976989/1.062894,8k .969246/1.187936.
Running-statistics U1 AUC/NLL:2k .978278/2.696895,8k .979023/2.680427.
The8k U1-minus-native accuracy gap is10.634766 points with running statistics,
versus6.767578 with batch statistics. Switching to checkpoint running statistics
therefore does not explain away the native/readout gap. Under running statistics,
native accuracy declines5.458984 points while U1 improves1.083984 points across
the nominated interval. Keep both protocols visible; do not replace the primary
batch-statistics trajectory or cherry-pick this protocol to resurrect the failed
original hidden-improvement hypothesis.

This broad qualitative pattern survives the normalization choice. The temporal
component crossover itself has still only been established under batch
statistics: this diagonal-only comparison does not prove encoder/inference
crossing effects persist with running statistics. It also does not rule out
other normalization-mediated mechanisms or justify target-based mode selection.

## Second-initialization replication decision, before its target outcomes

The seed-one pipeline now includes the fixed2k/4k/8k trajectory,2x2 crossover,
and normalization sensitivity after the500-episode8k readout and exact-initial
controls. Conditions and rules match seed zero; no additional target or repair
search. The crossover replication prediction is the observed seed-zero direction:
late inference helps both encoders and late encoder hurts both native inference
modules. This differs from the original rejected discovery prediction and is
recorded explicitly with `--replication` in the new protocol. Do not retrospectively
describe the seed-zero direction as predicted.

Classify outcomes separately: endpoint pretraining/readout gains may replicate
even if temporal decline or component effects do not. A failed component
replication limits that mechanism claim; it must not be hidden by averaging
seeds or by success of the endpoint result. The test remains conditional on
this target/recipe and does not alone establish general graph-model behavior.
No seed-one target results were available when this decision was written.

## Local temporal evidence export

The `data/publickg_{trajectory,crossover,normalization}_20260907/` folders retain
unaltered Tucker `summary.json`, `protocol.json`, and `execution_status.json`.
Summary hashes were compared against Tucker after copying. Run
`verify_temporal_evidence.py` in this analysis folder to verify those pinned
summary hashes, condition inventories,128 paired ordinals per condition, and all
means recomputed from episode rows, and to print the paper's crossover and
normalization tables. This check passed on the exported evidence.

Scope is deliberately limited: the large saved prediction/representation tensors
remain on Tucker. Their receipts were verified there before export; the local
check does not rerun inference or treat recurring query occurrences as independent
samples. The export provides auditable numerical evidence, not a new experiment.
