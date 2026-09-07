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

This unification is currently a hypothesis supported mechanistically on one
target/source pair. The next decisive test is whether transfer-health statistics
predict rankings across all available sources, targets, mixture schedules, and
seeds. If they do, the method contribution is a support-anchored adaptive GFM
that routes, fuses, or exits at the healthiest representation depth.

## Validity boundaries

- Episode-bootstrap intervals quantify paired variation over the 128 sampled
  episodes; they are not checkpoint-seed confidence intervals.
- The source pair was chosen for a large aggregate gap, so effect-size claims
  need replication on other pairs.
- This localizes where errors arise but does not yet identify which source
  training examples or gradients cause the deformation.
- Facebook is unusually favorable to raw features. Cross-target evaluation is
  necessary before making a general method claim.

