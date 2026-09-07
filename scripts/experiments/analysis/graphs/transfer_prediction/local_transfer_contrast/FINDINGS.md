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

## Framework seed: support-anchored transfer health

The localization supplies an actionable signal. At inference time, fit a cheap
readout from the episode support set at the raw input and optionally at each
intermediate layer. Treat agreement with the raw support-induced decision and
stability along the readout trajectory as a transfer-health measurement.

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

## Transfer health predicts which sources work

U1 agreement is useful beyond routing. Across the 45 singleton-source × target
cells, target-centered U1 agreement strongly correlates with transfer on the
fresh stream: Spearman rho is .740 for accuracy (p=6.1e-9) and .756 for AUC
(p=2.0e-9). The relationship appears independently on every substantive target:

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

As a global label-free source selector, choosing the model with highest fresh
U1 agreement improves accuracy over discovery-AUC selection on COVID, Election,
and Facebook, ties it on TwiBot, and fails on the chance-level suspended target.
Across the four substantive targets the mean gain is +.007 accuracy; mean AUC is
essentially unchanged because COVID accuracy improves while its AUC falls.
Health is already a strong ranking signal, but not yet a complete source selector.

This also sharpens the descriptive UKR/COVID observation. Across targets, the
Ukraine checkpoint has the highest mean accuracy/AUC and high U1 preservation
(.868); COVID has the highest mean preservation (.879) and is also strong. The
Facebook-source checkpoint has both low preservation (.669) and weak transfer.
Graph size or collection window may help create these weights, but neither is
the proximal explanation of their predictions: the measurable model-target
interaction is whether source training preserves a support-decodable decision
through final inference.

## Paper-level interpretation

The working thesis is **pretraining transfer is governed by preservation of
target support geometry, not global graph similarity alone**. This gives a
single interpretation of earlier observations:

- asymmetric transfer: source training deforms different target directions;
- more graphs are not always better: additional updates can destroy useful
  directions even when they add data;
- sequential mixtures underperform interleaving: long source blocks permit
  source-specific destructive drift, while interleaving can regularize it;
- adding another graph can improve a target: it can restore or preserve a target
  decision direction absent from the original source.

The matched-example mechanism is currently established on one target/source
pair, while the routing effect replicates across nine sources and five targets.
The next decisive tests are checkpoint-seed replication and mixture-schedule
trajectories. If health deteriorates under sequential merging and is preserved
by interleaving, the method contribution becomes a support-conditioned adaptive
GFM that routes, fuses, or exits at the healthiest representation depth—and the
same measurement explains the earlier training-order result.

## Validity boundaries

- Episode-bootstrap intervals quantify paired variation over the 128 sampled
  episodes; they are not checkpoint-seed confidence intervals.
- The mechanistic source pair was chosen for a large aggregate gap. The routing
  result now covers all nine available singleton sources but only seed-0 weights.
- This localizes where errors arise but does not yet identify which source
  training examples or gradients cause the deformation.
- Agreement is not sufficient when the intermediate readout is itself at chance;
  the health score needs a competence term on low-signal targets.
- Facebook is unusually favorable to raw features. Cross-target evaluation is
  why raw anchoring is not the general method; U1 health is the cross-target rule.
