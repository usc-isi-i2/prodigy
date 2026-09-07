# Local transfer contrast: source-induced decision deformation

## Question and fixed contrast

Why does one pretrained graph model transfer better than another on the same
target examples? We selected the contrast from the three-seed final-core matrix
before inspecting individual outcomes:

- target: Facebook page-reference;
- stronger source: TwiBot-20 (mean transfer ROC-AUC .808 in the matrix);
- weaker source: Election-2020 (mean .664).

Both fixed checkpoints consume bit-identical target episodes. The original
1,024 query occurrences are discovery data; a separately sampled 1,024-query
stream is validation. The validation stream is not a new checkpoint seed or a
held-out domain.

## Findings

### The global ranking hides substantial local complementarity

On validation, TwiBot-20 reaches .681 accuracy/.754 AUC and Election-2020
reaches .600/.657. Nevertheless, Election alone is correct on 175 queries that
TwiBot misses; TwiBot alone is correct on 258. Their per-query oracle is .852
accuracy. Discovery gives essentially the same decomposition (265/162 unique
wins and .853 oracle accuracy).

Raw cross-model losses initially give the wrong impression because the two
checkpoints have very different logit scales. Temperatures fitted on discovery
are 3.49 (TwiBot) and 2.57 (Election). All loss comparisons use these frozen
temperatures.

### Raw target location does not identify the winning source

Input-only clusters were selected by silhouette score without outcome labels
and frozen for validation. Clusters over the center bio embedding, sampled
neighbor mean, their concatenation, and structural metadata do not preserve a
source-ranking reversal. Logistic routers trained diagnostically on discovery
outcomes from these inputs also fail to beat always choosing TwiBot; the best
input-only validation accuracy is .648. This first contrast does not support a
simple “the weak model wins on a distant region of p(x)” account.

### The ranking is created after the shared input

The raw-feature prototype/ridge logits are bit-identical between checkpoints.
A support-fitted ridge readout on center + neighbor features reaches .832
accuracy/.904 AUC on validation, well above both pretrained models. Performance
then falls as source-specific computation is introduced:

| Readout | TwiBot-20 | Election-2020 |
|---|---:|---:|
| raw joint / ridge | .832 / .904 | .832 / .904 |
| pre-metagraph encoder / ridge | .713 / .799 | .686 / .754 |
| post-metagraph / ridge | .647 / .739 | .588 / .619 |
| full model | .681 / .754 | .600 / .657 |

Values are validation accuracy/AUC. Thus the target contains strong usable
evidence before pretraining-specific computation, and the transfer ranking
emerges through unequal destruction of that evidence.

Conditioning on the 175 validation queries where only Election is correct makes
this concrete. The common raw readout is correct on 74.9%. TwiBot's encoder
breaks 50.9% of raw-correct decisions, whereas Election's breaks 16.6%. On the
258 TwiBot-only queries, raw is correct on 86.8%; Election breaks 39.5% and
TwiBot breaks 7.8%. The asymmetry is not that one source owns all useful target
examples. Each checkpoint preserves a different subset of an initially useful
decision geometry.

### Individual examples agree with the localization

We selected 24 distinct loss-extreme query IDs before reading text, reconstructed
their exact cached query/support subgraphs, and verified all 629 hydrated text
embeddings against the graph tensors. Many flips are not semantically obscure.
A Liberal Democrats page surrounded by political-party supports, a Virginia
senator surrounded by politicians, and a church-learning page surrounded by
religious organizations are all correctly separated by the raw support readout.
For each, one source encoder flips the correct raw decision while the other
preserves it. Hydrated text and identifiers remain in ignored Tucker output and
are not committed.

## Framework: TRACE

The localization supplies an actionable signal. We call the resulting framework
**TRACE: Transfer Readout Agreement and Competence Estimation**. At inference
time, TRACE fits cheap support-set readouts at intermediate layers and asks two
questions: can the representation recover held-out supports, and does the final
model preserve its support-induced query decision? The first measures competence;
the second measures source-target stability.

A label-free two-expert rule already works: when the fixed experts disagree,
choose the one that agrees with the raw support-fitted readout; otherwise retain
the stronger expert. On validation this reaches .775 accuracy/.821 AUC, a +.095
accuracy gain over TwiBot (paired episode-bootstrap 95% interval [.070, .119]).
Using consensus across learned-stage readouts reaches .748/.798, +.067
[.047, .089]. Direct raw ridge reaches .832/.904 but has worse NLL, suggesting
that adaptive routing or calibrated fusion is preferable to discarding the
pretrained path outright.

A target-query-supervised logistic router over all readout trajectories reaches
.701/.781 on validation and .700 accuracy on validation users absent from the
discovery stream. It is a diagnostic upper bound, not a valid final method; the
label-free consistency rules are the deployable evidence.

## All-source, cross-target replication

We next exported all nine singleton-source checkpoints for every target in the
verified role-context replay. For each target, experts are ranked once by AUC on
the discovery stream. On every validation query, **U1 health routing** selects
the highest-ranked expert whose full prediction agrees with a support-fitted
ridge readout at the pre-metagraph representation; if none qualifies, it uses
the highest-ranked expert. Routing uses support labels and model outputs, never
the validation query label.

| Target | Best fixed | U1 health | Gain [episode-bootstrap 95% interval] | AUC fixed → health |
|---|---:|---:|---:|---:|
| COVID-political | .896 | .914 | +.019 [.013, .024] | .957 → .954 |
| Election-2020 | .973 | .977 | +.004 [.000, .012] | .984 → .984 |
| Facebook page-reference | .681 | .715 | +.034 [.019, .051] | .754 → .778 |
| TwiBot-20 | .586 | .632 | +.047 [.035, .059] | .615 → .643 |
| Ukraine/Russia suspended | .531 | .562 | +.031 [-.016, .078] | .506 → .548 |
| **Macro mean** | **.733** | **.760** | **+.027** | **.763 → .781** |

The method improves accuracy on all five targets. Four point estimates improve
with nonnegative paired intervals; the small 256-query suspended target is
directionally positive but inconclusive. NLL improves on the four substantive
targets, including COVID where AUC falls by .003. The method routes away from
the top expert on only 0.4%–32.4% of examples depending on target, so it behaves
as a selective correction rather than an ensemble replacement.

This replication also falsifies the literal raw-anchor method. Raw agreement is
excellent on Facebook but harms COVID, Election, and TwiBot. The transferable
principle is therefore **support-conditioned consistency at an appropriate
intermediate depth**, not privileged status for raw features.

## Transfer health predicts which sources work across checkpoint seeds

We reran all nine singleton sources on all five targets for all three independently
trained final-core checkpoint seeds. Target episodes and sampled subgraph tensors
are fixed exactly across checkpoint seeds. Across the resulting 135 fresh
source-target-seed cells, target-and-seed-centered U1 agreement strongly correlates
with transfer: Spearman rho is .710 for accuracy (p=5.5e-22) and .698 for AUC
(p=5.0e-21). The discovery stream independently gives rho=.738/.722. Thus the
relationship is not a single-checkpoint artifact.

On the seed-0 slice, the relationship appears independently on every substantive
target:

| Target | rho with accuracy | rho with AUC |
|---|---:|---:|
| COVID-political | 1.000 | .950 |
| Election-2020 | .889 | .765 |
| Facebook page-reference | .733 | .733 |
| TwiBot-20 | .583 | .650 |
| Ukraine/Russia suspended | -.332 | .050 |

The exception is informative: on the suspended benchmark, all sources are near
chance, so agreement can reflect stable but uninformative computation. Transfer
health should therefore be combined with minimum support-readout competence,
not interpreted as quality in isolation.

As a global query-label-free source selector, choosing the model with highest fresh
U1 agreement improves accuracy over discovery-AUC selection on COVID, Election,
and Facebook, ties it on TwiBot, and fails on the chance-level suspended target.
Across the four substantive targets the mean gain is +.007 accuracy; mean AUC is
essentially unchanged because COVID accuracy improves while its AUC falls.
Health is already a strong ranking signal, but not yet a complete source selector.

This also sharpens the descriptive UKR/COVID observation. Averaged over target
and checkpoint seed, the Ukraine source has the highest fresh mean accuracy/AUC
(.730/.764) and the highest mean U1 preservation (.877); COVID is also strong
(.716/.754, preservation .875). The Facebook-source checkpoints have both low
preservation (.683) and weak transfer (.632/.669).
Graph size or collection window may help create these weights, but neither is
the proximal explanation of their predictions: the measurable model-target
interaction is whether source training preserves a support-decodable decision
through final inference.

## Query-label-free routing and an honest scope test

The initial U1 router uses a target discovery split to rank experts before applying
the label-free health gate. TRACE removes that dependency:

1. For each expert and episode, compute leave-one-support-out prototype accuracy
   at U1. This uses only the 20 labeled supports already supplied to the model.
2. For each query, retain experts whose full prediction agrees with their
   support-fitted U1 prediction.
3. Among retained experts, select the one with the highest episode support
   competence; break ties by mean unlabeled target agreement.
4. Abstain on the target if no expert's mean support competence exceeds .55,
   a fixed chance+.05 threshold.

The threshold covers COVID-political, Election-2020, Facebook page-reference,
and TwiBot-20 for every checkpoint seed, while rejecting Ukraine/Russia suspended
for every seed. Covered-target competence is at least .629; suspended competence
is .494--.501. This precisely detects the benchmark on which layer agreement is
stable but uninformative.

Against the stronger baseline that selects one fixed expert using labeled target
discovery AUC, query-label-free TRACE improves covered-target macro accuracy from
.792 to .812 (+.021) and macro AUC from .834 to .835. Accuracy improves in 10/12
target-seed cells and never degrades; Election is tied in two seeds. The effect is
largest and fully seed-consistent on the three non-ceiling targets:

| Target | Fixed → TRACE accuracy | Mean gain | Seed range |
|---|---:|---:|---:|
| COVID-political | .900 → .921 | +.021 | +.014 to +.027 |
| Election-2020 | .975 → .977 | +.001 | .000 to +.004 |
| Facebook page-reference | .683 → .726 | +.043 | +.025 to +.059 |
| TwiBot-20 | .608 → .626 | +.018 | +.002 to +.037 |

For the rejected suspended target, the forced router would lose .042 accuracy on
average. The abstention rule therefore is part of the result, not cosmetic
filtering: support competence identifies when the model family lacks transferable
signal and prevents TRACE from presenting a meaningless routing claim.

## Schedule intervention: health tracks effects, not a universal winner

The eight-source 40k ladder previously showed a strong schedule effect on its NM
objective: sequential minus interleaved AUC averages -.044 across 64 paired cells,
including -.070 on incumbent graphs, while the newest graph averages +.006. A
separate retained two-source pilot instead showed target-dependent effects. We
therefore ran a controlled three-seed study at two, three, and four sources rather
than promote either pattern into a universal schedule rule.

The 27 new models hold total updates, exact source exposure, anchors, members,
support/query roles, sampled context nodes, and context edges fixed across blocked,
100-episode replay, and one-episode interleaving. The predeclared pair-to-many
interaction does not replicate: on the fresh support-competent targets it is
-.0001 accuracy (crossed seed × target 95% interval [-.0151, .0119]) and -.0014
AUC ([-.0190, .0167]). Thus source count alone is not a universal switch that
makes blocked training fail; the historical result may require its longer horizon,
eight-source scale, budget, order, or sampler.

The mechanistic result does replicate. Across 72 above-chance schedule contrasts,
the change in U1 agreement tracks the change in accuracy at rho=.585
([.159, .826]) on the fresh stream. The independent original stream gives
rho=.548 ([.101, .897]). Schedule changes that preserve target support geometry
better tend to change target accuracy in the same direction even when their
training examples are exact-matched.

This also enables an untuned deployment rule. TRACE health fusion averages the
probabilities of schedule checkpoints whose final prediction agrees with their U1
support readout, falling back to all three if none agree. On four supported targets
it improves a stronger labeled-discovery-selected fixed checkpoint from .7851 to
.7989 accuracy (+.0138, crossed interval [+.0036, +.0237]) and from .8296 to
.8333 AUC (+.0037, interval [-.0019, +.0113]). The accuracy gain appears in 27/36
cells, with six ties and three losses. The .55 support-competence gate rejects the
chance-level suspended target in all nine cells. Full details are in
`../trace_schedule_scaling/FINDINGS.md`.

## Paper-level interpretation

The working thesis is **pretraining transfer is governed by preservation of
target support geometry, not global graph similarity alone**. This follows the
productive pattern used by data-active graph pretraining work: turn the failure
of naive scaling into a measurable model-data interaction and then use that
measurement to control computation. Here the interaction is representation
survival rather than graph size or predictive uncertainty. It gives a single
interpretation of earlier observations:

- asymmetric transfer: source training deforms different target directions;
- more graphs are not always better: additional updates can destroy useful
  directions even when they add data;
- the eight-source sequential ladder underperforms interleaving mainly on incumbent
  graphs, consistent with destructive drift, while the short two-source pilot shows
  that the direction is not universal;
- adding another graph can improve a target: it can restore or preserve a target
  decision direction absent from the original source.

The matched-example mechanism is established on one target/source pair, while
health predictiveness and routing replicate across nine sources, five targets,
three checkpoint seeds, and an exact-matched schedule intervention. The evidence
supports transfer diagnosis, abstention, and adaptive inference. A retained
eight-source trajectory and a health-preserving training regularizer remain the
next scaling tests rather than prerequisites for the present mechanism claim.

## Validity boundaries

- Episode-bootstrap intervals quantify paired variation over the 128 sampled
  episodes; they are not checkpoint-seed confidence intervals.
- The mechanistic source pair was chosen for a large aggregate gap. Predictiveness
  and routing cover all nine singleton sources and three checkpoint seeds, but the
  layer-by-layer matched-example localization is still one target/source contrast.
- This localizes where errors arise but does not yet identify which source
  training examples or gradients cause the deformation.
- Agreement is not sufficient when the intermediate readout is itself at chance;
  TRACE's competence threshold abstains in this setting. The .55 threshold is
  principled relative to binary chance but has not yet been varied.
- Facebook is unusually favorable to raw features. Cross-target evaluation is
  why raw anchoring is not the general method; U1 health is the cross-target rule.
- The controlled schedule study reaches four sources and 2,500 updates; it limits
  the generality of the old eight-source/40k effect but does not reproduce that
  scale or optimization horizon.
