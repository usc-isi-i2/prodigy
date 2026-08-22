# Strict GraphSAGE transfer study

## Purpose

This document records the stricter GraphSAGE experiments developed after the
initial fixed-2,500-step mixture ladder. The immediate question is whether an
encoder pretrained on a mixture of non-target graphs transfers better than a
target specialist, a single-source encoder, an untrained GraphSAGE encoder, and
simple feature baselines. It also records the structural-feature follow-up on
TwiBot-20 and Ukraine/Russia suspended.

These experiments are a first step toward the broader research questions:

1. How does downstream performance change with pretraining-mixture scale and
   diversity?
2. How does that relationship change with model scale?
3. Can the observed relationships predict a compute-efficient mixture that
   outperforms naive mixture choices?

The study uses plain GraphSAGE throughout. PRODIGY, episodic learning, label
embeddings, pooling supernodes, and metagraph attention are not used.

## Graphs

The seven graphs are:

- COVID political
- Ukraine/Russia suspended (UKR/RUS)
- Election 2020
- TwiBot-20
- Facebook page reference
- Cora
- PubMed

For a target-held-out six-source mixture, the target graph is excluded and the
other six graphs are the SSL sources. Sampling rotates uniformly across sources;
it is not proportional to graph size.

## Permanent data splits and leakage control

Each graph has one permanent, stratified node split made with split seed 1729:
70% train, 15% validation, and 15% test. An induced graph is constructed separately
for each partition, so no SSL edge, sampled neighborhood, or structural statistic
crosses partition boundaries.

SSL uses only train induced graphs for gradient updates. SSL checkpoint selection
uses only validation induced graphs. Target labels are not read during pretraining.
Downstream hyperparameters are selected using labeled train and validation nodes;
the selected classifier is refit on their union and scored once on labeled test
nodes. Test labels never select a checkpoint or classifier.

Known labeled partition sizes are:

| Target | Train | Validation | Test |
|---|---:|---:|---:|
| TwiBot-20 | 8,277 | 1,773 | 1,776 |
| UKR/RUS | 39,505 | 8,465 | 8,468 |

The corresponding total node partitions are 114,091/24,447/24,452 for TwiBot-20
and 50,604/10,843/10,848 for UKR/RUS. Unlabeled nodes participate in SSL and
message passing but not classifier fitting or classification metrics.

## Encoder and SSL method

The encoder is a one-layer GraphSAGE model with 256 hidden and 256 output
dimensions, no dropout, and fanout 15. The native SSL objective predicts observed
source-confined edges against five sampled negatives per positive. A batch contains
1,024 examples. Mixture minibatches contain exactly one graph and rotate across
sources, preventing cross-graph positive or negative pairs.

Optimization uses AdamW with learning rate 0.001 and weight decay 1e-5. Training
allows at most 10,000 updates. Validation is measured every 250 updates over 20
batches per source. Runs train for at least 2,500 updates and stop after five
validation checks without the configured 0.2% relative improvement. Checkpoints
are retained throughout training, but the single absolute-lowest SSL validation
loss checkpoint is used for final downstream evaluation. A post-training audit
checks source identity, split hashes, source confinement, seed, and selected loss.

The initial seed is 0. The main TwiBot existing-feature comparison was subsequently
repeated at seeds 1 and 2 for the target specialist, Election 2020 source, and
six-source mixture.

### Validation caveat

The induced 15% validation graph is much smaller and can have a materially different
edge-density distribution from the 70% training graph. This is particularly visible
for Election 2020, whose validation loss rises while its training loss falls. Mixture
validation is consequently noisy and sometimes dominated by one source. Results
below are valid under the declared selection protocol, but a definitive scaling
study should select SSL convergence using held-out edges within each training graph
rather than a separately induced validation-node graph.

## Downstream evaluation

The primary downstream statistic is macro one-vs-rest ROC AUC. Frozen evaluation
embeds nodes with a fixed encoder and fits logistic regression. Regularization
strength C is selected from 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, and 100 using validation
AUC. The chosen head is refit on labeled train plus validation nodes, then evaluated
once on the test partition. Macro F1, accuracy, and balanced accuracy are saved as
secondary metrics, but AUC is the analysis metric reported here.

"Scratch GraphSAGE" means the same randomly initialized encoder followed by the
same frozen-probe procedure; it is not a supervised model. The supervised
GraphSAGE baseline trains the encoder directly on target train labels, selects the
epoch with target validation AUC, and then tests the selected model once.

## Existing-feature TwiBot-20 results

All rows below use the original 768-dimensional node features and a frozen linear
probe. Three-seed means are available for the four central conditions.

| Encoder pretraining | Seed 0 AUC | Seed 1 AUC | Seed 2 AUC | Mean AUC |
|---|---:|---:|---:|---:|
| Scratch (none) | 0.7205 | 0.7081 | 0.6965 | **0.7084** |
| Election 2020 | **0.7231** | 0.7022 | **0.6972** | 0.7075 |
| Six-source held-out mixture | 0.7073 | 0.6889 | 0.6908 | 0.6957 |
| TwiBot-20 target SSL | 0.6915 | 0.6870 | 0.6879 | 0.6888 |

Additional seed-0 single-source transfer AUCs are:

| SSL source | Test AUC |
|---|---:|
| Cora | 0.7032 |
| PubMed | 0.7030 |
| COVID political | 0.7004 |
| UKR/RUS | 0.6909 |
| Facebook page reference | 0.6781 |

Logistic regression directly on the 768 existing features scores 0.7046 AUC.
Therefore, on TwiBot-20, the existing-feature GraphSAGE SSL encoders do not beat
the best scratch result or establish a mixture advantage. The mixture behaves more
like an average source than a best-source selector in this small comparison.

## Structural features

Sixteen graph statistics are computed independently inside each induced partition:
log in-degree, log out-degree, log total directed degree, in/out ratio, reciprocal
fraction, log undirected degree, clustering coefficient, log triangle count, log
k-core number, log mean/standard-deviation/maximum neighbor degree, log two-hop walk
count, two-hop expansion, egonet density, and scaled PageRank. Their within-partition
percentile ranks are appended, producing 32 structural features.

For logistic baselines, these are used alone or concatenated with the 768 existing
features. For GraphSAGE, the 32 structural columns are z-scored using statistics from
each source's training partition; those fixed statistics are then applied to that
source's validation partition. Target train statistics are likewise fixed for target
validation and test, giving 800 input dimensions. Earlier unnormalized GraphSAGE
structural runs are diagnostic only and are excluded from reportable comparisons.

## Structural follow-up: TwiBot-20

All entries are seed 0 and use the same permanent split.

| Method/input | Test AUC |
|---|---:|
| Structural-only logistic regression | 0.6993 |
| Existing-only logistic regression | 0.7046 |
| Scratch GraphSAGE, structural + existing | 0.7117 |
| Target structural SSL + frozen probe | 0.7457 |
| Supervised GraphSAGE, structural + existing | 0.7463 |
| Structural + existing logistic regression | **0.7540** |

Normalized target SSL improves 3.40 AUC percentage points over its paired scratch
encoder and nearly matches supervised GraphSAGE. However, the simple concatenated
logistic baseline remains best by 0.77 points. This shows that structural information
is highly useful on TwiBot-20, while the current GNN does not yet improve upon a
linear classifier using the same information.

## Structural follow-up: UKR/RUS

All entries are seed 0 and use the same permanent split.

| Method/input | Test AUC |
|---|---:|
| Structural-only logistic regression | 0.5358 |
| Existing-only logistic regression | 0.6048 |
| Target structural SSL + frozen probe | 0.6068 |
| Scratch GraphSAGE, structural + existing | 0.6106 |
| Six-source structural mixture + frozen probe | 0.6106 |
| Supervised GraphSAGE, structural + existing | 0.6128 |
| Structural + existing logistic regression | **0.6162** |

The six-source structural mixture excludes UKR/RUS and contains COVID political,
Election 2020, TwiBot-20, Facebook page reference, Cora, and PubMed. It improves
over target-only structural SSL by 0.37 AUC points, is effectively tied with its
paired scratch encoder, and trails concatenated logistic regression by 0.56 points.

A matched held-out mixture using only the 768 existing features scores 0.5823 AUC.
Adding structural features raises mixture AUC to 0.6106, an improvement of **2.82
percentage points**. This is the cleanest current evidence that the mixture benefits
from structural inputs, although it is only one seed and one target.

## Current conclusions

1. Pretraining source matters, but effects are small and target-dependent with the
   existing features; on TwiBot-20 the observed range is roughly 4.5 AUC points and
   the mixture is not the best encoder.
2. More mixture sources are not monotonically better in the earlier seven-target
   ladder. Source identity remains confounded with size under a single addition order.
3. Structural features materially improve held-out mixture transfer on UKR/RUS and
   target SSL on TwiBot-20.
4. Strong feature-only logistic baselines are essential. They currently match or
   beat GraphSAGE on both structural targets.
5. The present evidence does not yet establish a general scaling law or answer the
   model-scale and compute-efficient-design questions.

## Recommended next experiments

- Replace induced-validation-node SSL selection with held-out training-edge
  validation and repeat the matched structural/no-structural comparisons.
- Repeat UKR/RUS structural mixture, target specialist, scratch, and logistic
  baselines for at least two additional seeds.
- Run structural and existing-feature comparisons across all seven held-out targets.
- At each mixture size, sample multiple source subsets/orders so mixture scale can
  be separated from source identity and graph size.
- Only after the RQ1 protocol is stable, repeat the ladder at several GraphSAGE
  widths/depths for RQ2 and fit held-out predictive mixture-selection tests for RQ3.

## Artifact locations

Versioned code, configs, tests, aggregate tables, and figures live in this repository.
Raw graphs, split files, checkpoints, JSON result shards, and logs remain on Tucker
under `/dataMeR1/phil/gfm/mixture-scaling`. Existing and interrupted runs were
preserved rather than deleted.
