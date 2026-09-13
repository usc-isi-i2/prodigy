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

SSL uses only train induced graphs. Within every source training graph, a fixed,
deterministic 10% of positive edges is removed from message passing and gradient
updates and used only for SSL checkpoint selection. The remaining 90% supplies SSL
training positives. Thus training and validation edges are disjoint while both come
from the same node-induced graph; node validation and test partitions remain
entirely untouched by SSL. Target labels are not read during pretraining.
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

### Superseded validation pilot

The earliest strict runs selected SSL checkpoints on the separately induced 15%
validation-node graph, introducing a graph-density shift most visibly on Election
2020. Reportable scaling runs now use held-out training edges. A matched TwiBot
six-source rerun reproduced the endpoint conclusion: existing-feature AUC changed
from 0.7073 to 0.7058 and structural-feature AUC from 0.7558 to 0.7560.

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

## Complete seven-target seed-0 benchmark

The requested seven-target by eight-method table is complete, with one final test
evaluation per fitted model. The compact table below omits structural-only logistic
regression; its AUCs are 0.6993, 0.5358, 0.5099, 0.9373, 0.5251, 0.4923, and 0.5199
in table order.

| Target | Existing LR | Combined LR | Scratch SAGE | Target SSL | Mixture existing | Mixture structural | Supervised SAGE |
|---|---:|---:|---:|---:|---:|---:|---:|
| TwiBot-20 | 0.7046 | 0.7540 | 0.7117 | 0.7457 | 0.7073 | **0.7558** | 0.7463 |
| UKR/RUS | 0.6048 | **0.6162** | 0.6106 | 0.6068 | 0.5823 | 0.6106 | 0.6128 |
| COVID political | **0.9127** | 0.9088 | 0.8446 | 0.8634 | 0.8870 | 0.8628 | 0.9026 |
| Election 2020 | 0.9536 | 0.9723 | **0.9816** | 0.9725 | 0.9776 | 0.9803 | 0.9815 |
| Facebook reference | **0.9097** | 0.9096 | 0.8912 | 0.8949 | 0.8934 | 0.8918 | 0.9053 |
| Cora | **0.9787** | 0.9783 | 0.9615 | 0.9626 | 0.9568 | 0.9577 | 0.9580 |
| PubMed | 0.9873 | **0.9874** | 0.9791 | 0.9795 | 0.9812 | 0.9790 | 0.9857 |

## Corrected mixture scale and diversity experiment

The held-out-edge ladder contains 112 seed-0 evaluations: every target, mixture
sizes 1 through 6, three deterministic source subsets at sizes 1--5, and the unique
six-source mixture. Feature mode was fixed per target before this sweep from the
three-seed endpoint evidence (structural for TwiBot, UKR/RUS, and Election; existing
for the other four).

| Target | k=1 | k=2 | k=3 | k=4 | k=5 | k=6 | Best k |
|---|---:|---:|---:|---:|---:|---:|---:|
| TwiBot-20 | 0.7305 | 0.7465 | 0.7514 | 0.7519 | 0.7533 | **0.7560** | 6 |
| UKR/RUS | 0.6085 | 0.6089 | 0.6068 | **0.6102** | 0.6087 | 0.6061 | 4 |
| COVID political | 0.9015 | 0.8749 | 0.8899 | **0.9019** | 0.8643 | 0.8983 | 4 |
| Election 2020 | **0.9813** | 0.9790 | 0.9810 | 0.9774 | 0.9758 | 0.9788 | 1 |
| Facebook reference | 0.8938 | **0.8940** | 0.8924 | 0.8909 | 0.8935 | 0.8932 | 2 |
| Cora | **0.9619** | 0.9582 | 0.9600 | 0.9589 | 0.9578 | 0.9593 | 1 |
| PubMed | 0.9806 | 0.9790 | 0.9797 | 0.9802 | 0.9803 | **0.9811** | 6 |

The target-centered pooled slope against log mixture size is 0.00057 AUC
(p=0.733): there is no general monotone scaling law. Only TwiBot shows a clear
positive relationship (+2.55 points from k=1 to k=6). Source-subset standard
deviation averages roughly 0.3--0.6 points and commonly exceeds the mean effect of
adding sources. A coarse same-domain versus mixed-domain contrast is also null
(mean +0.18 points, p=0.472). Source identity and target compatibility dominate
naive mixture size.

Across three seeds, a cheap target-only diagnostic predicts whether structural
mixture inputs help: combined-minus-existing logistic-regression gain correlates
with structural-minus-existing mixture gain (Pearson r=0.915, p=0.0039). The rule
“use structural inputs iff the cheap logistic gain is positive” selects the better
mean mixture mode on six of seven targets and averages 0.8630 AUC, versus 0.8523
for always-existing, 0.8605 for always-structural, and 0.8640 for an oracle. This is
preliminary RQ3 evidence and requires held-out validation across more graphs.

## Model scale interaction

RQ2 fixes the input representation to the original 768 node features and varies the
one-layer GraphSAGE output width over 64, 256, and 512. For every target and width,
it evaluates the exact same three source subsets at k=1 and k=3 and the unique k=6
mixture. The complete audit contains 147 cells, including paired width-256 rows
reused from the corrected ladder where applicable.

| Target | k=6 minus k=1, width 64 | width 256 | width 512 |
|---|---:|---:|---:|
| TwiBot-20 | +0.0084 | -0.0029 | -0.0098 |
| UKR/RUS | -0.0037 | -0.0024 | -0.0075 |
| COVID political | **-0.0643** | -0.0032 | -0.0021 |
| Election 2020 | +0.0098 | +0.0050 | +0.0031 |
| Facebook reference | +0.0089 | -0.0006 | -0.0001 |
| Cora | +0.0019 | -0.0026 | -0.0000 |
| PubMed | -0.0054 | +0.0005 | +0.0002 |

Increasing width improves absolute AUC on every target, but does not induce a
universal mixture-scaling law. The pooled log-width by log-mixture-size interaction
is 0.00102 AUC (p=0.590). Instead, width changes target-specific transfer and
interference. The clearest example is COVID: the six-source mixture loses 6.43 AUC
points relative to k=1 at width 64, but only 0.21--0.32 points at widths 256--512.
Election shows the opposite pattern: extra sources help at every width, most at
width 64. TwiBot switches from a positive k=6 effect at width 64 to a negative one
at width 512. These sign changes rule out a single global capacity correction to
mixture scale.

The COVID interaction was then replicated at seeds 1 and 2. The k=6-minus-k=1
effects at width 64 are -0.0643, -0.0611, and -0.0774 across seeds; at width 512
they are -0.0021, -0.0082, and -0.0205. Mean mixture interference is therefore
-0.0676 +/- 0.0086 AUC at width 64 versus -0.0103 +/- 0.0094 at width 512. Every
seed shows the same capacity mitigation, averaging +0.0573 AUC. This confirms that
the COVID small-model collapse is not a seed-0 anomaly.

The TwiBot sign change was also replicated at seeds 1 and 2. Its k=6-minus-k=1
effects are +0.0084, +0.0025, and +0.0071 at width 64, but -0.0098, -0.0035, and
-0.0076 at width 512. The three-seed means are +0.0060 +/- 0.0031 and
-0.0070 +/- 0.0032, respectively. Unlike COVID, increasing capacity makes the
mixture effect more negative; every seed shows the same directional change. Taken
together, COVID and TwiBot establish that capacity can either suppress or induce
mixture interference depending on the target.

## Current conclusions

1. Pretraining source matters, but effects are small and target-dependent with the
   existing features; on TwiBot-20 the observed range is roughly 4.5 AUC points and
   the mixture is not the best encoder.
2. More mixture sources are not generally better in the corrected multi-subset
   seven-target ladder. Source identity and target compatibility dominate scale.
3. Structural features materially improve held-out mixture transfer on UKR/RUS and
   target SSL on TwiBot-20.
4. Strong feature-only logistic baselines are essential. They currently match or
   beat GraphSAGE on both structural targets.
5. Model width improves absolute performance, but its interaction with mixture size
   changes sign by target; a single global joint scaling law is rejected.
6. A cheap target diagnostic nearly recovers oracle feature-mode selection, providing
   a concrete compute-efficient-design hypothesis.

## Recommended next experiments

- Validate the cheap feature-mode rule leave-one-target-out or on additional graphs.
- Model source compatibility from graph metadata and test selection against random,
  largest-first, and all-source mixtures at matched compute.

## Artifact locations

Versioned code, configs, tests, aggregate tables, and figures live in this repository.
Raw graphs, split files, checkpoints, JSON result shards, and logs remain on Tucker
under `/dataMeR1/phil/gfm/mixture-scaling`. Existing and interrupted runs were
preserved rather than deleted.
