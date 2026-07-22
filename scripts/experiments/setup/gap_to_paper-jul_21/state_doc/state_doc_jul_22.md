Does “neural scaling laws for GNNs” hold for us?
Can we use our representations for autoregressive text gen?

word2vec results for graphs:
- they found king-royal=man. can we get somethign similar?
- can we get *controllable* representations this way?

!we could experiment with true merging instead of interleaving.

Contributions:
- Model — The first foundation model for social networks: one SSL encoder pretrained across 8 large feature-rich social graphs, transferring few-shot to held-out graphs and task families (bot, suspension, political leaning, node regression, link prediction) under linear probe and finetuning.
- Recipe: how to train on multi-domain and multi-objective.
- Scaling: how to scale to 100M data points—how to keep learning even after millions of episodes. The first controlled study of pretraining allocation over graph×objective pairs at matched compute: graph exposure (balanced vs proportional, within- vs cross-source episodes) crossed with objective combination (sequential vs rotation vs joint). Analyzed with a gain ledger separating target-membership, single-donor coverage, multi-source synergy, and dilution, plus gradient-conflict and forgetting measurements. [Promote if results land: a cheap-proxy rule that picks the right mixture at scale; the recipe turning the flat multi-graph scaling curve into a rising one.]
- Interaction: evidence that data and objective choices are non-separable (which objective mix wins depends on the corpus), so the recipe must be chosen jointly.


List
- Model — The first foundation model for social networks: one SSL encoder pretrained across 8 large feature-rich social graphs, transferring few-shot to held-out graphs and task families (bot, suspension, political leaning, node regression, link prediction) under linear probe and finetuning. [Could extend with: ICL evaluation; beats task-specific pipelines and generic GFMs (GFT/OFA).]
- Recipe — The first controlled study of pretraining allocation over graph×objective pairs at matched compute: graph exposure (balanced vs proportional, within- vs cross-source episodes) crossed with objective combination (sequential vs rotation vs joint). Analyzed with a gain ledger separating target-membership, single-donor coverage, multi-source synergy, and dilution, plus gradient-conflict and forgetting measurements. [Promote if results land: a cheap-proxy rule that picks the right mixture at scale; the recipe turning the flat multi-graph scaling curve into a rising one.]
- Non-separability — Data and objective allocations interact: which objective mix wins depends on the corpus (and vice versa), so the two must be tuned jointly — no prior work shows this. (Standalone contribution; your best-evidenced claim via the Jul 21 corpus inversion.)
- Mechanism — What actually transfers: objectives read different channels (feature content vs topology), a source's value depends on the objective applied to it, and feature-cloud divergence predicts when transfer works.
- Resource — Released weights plus the audited social-graph transfer benchmark: repaired pair evaluator, shortcut-aware floors, supervised skylines, event-family/identity-clean holdout map. (D&B companion paper as parallel track or fallback.)

List1
- Source-mixture policy for multi-graph pretraining — first study of exposure allocation (balanced vs proportional, within- vs cross-source episodes) at fixed compute, with a cheap-proxy rule that picks the right mixture at scale.
- Gain decomposition — a ledger separating target-membership, single-donor coverage, genuine multi-source synergy, and dilution; no prior work isolates these.
- Objective-combination mechanisms for transfer — sequential vs rotation vs joint loss at matched compute in the cross-graph setting (all prior work is single-graph), explained via gradient-conflict and forgetting measurements.
- Non-separability — the best objective mix depends on the data mix (and vice versa), so the two allocations must be tuned jointly; zero precedent.
- Artifact — the first SSL foundation model pretrained across multiple large feature-rich social graphs, plus the audited transfer benchmark (shortcut-aware floors, event-family/user holdouts), both released.

List2
- Recipe/method: first study of compute allocation over graph×objective pairs — showing it's non-separable and giving the allocation recipe (balanced within-source sampling + objective rotation + joint mixture) that turns the flat multi-graph scaling curve into a rising one.
<!-- - Model: the first open foundation model for social networks — one encoder, 8 graphs, transferring few-/zero-shot to unseen graphs across classification, regression, and link prediction. -->
- Mechanism: what actually transfers — objectives read different channels (feature content vs topology), and a source's value depends on the objective applied to it.
- Benchmark: the audited social-graph transfer suite — 8 graphs, event-family/identity-clean splits, repaired pair evaluator, floors + supervised skylines (potential parallel D&B paper).

List3
<!-- - Model: the first foundation model for social networks — one pretrained encoder transferring to held-out graphs and task families (bot, suspension, political leaning, regression, LP) under ICL, linear-probe, and finetuning, beating task-specific pipelines and generic GFMs (GFT/OFA). -->
<!-- - Recipe: a controlled study of the pretraining mixture on both axes — sequential vs interleaved vs joint for graphs and objectives — with sampling/exposure controls showing how to combine them without dilution or forgetting. -->
- Interaction: evidence that data and objective choices are non-separable (which objective mix wins depends on the corpus), so the recipe must be chosen jointly.
- Mechanism: ablations showing what the objectives actually use (feature content vs topology), explaining when transfer works and predicting it from feature divergence.
- Resource: released weights + an audited multi-graph social benchmark (valid pair evaluator, floors, holdout map) — also the D&B fallback.

<!-- TODO
is the goal for downstream tasks:
good 0-shot performance
minimize need for FT
get best performance after FT -->

Title: "How to Train Your GFM” — A Pretraining Recipe for social graph FMs

# Title
> We train the first foundation model for social networks.

### Abstract
The untuned lever in graph foundation-model pretraining is the curriculum: which source graphs and which objectives get the fixed compute budget, and how they're interleaved. We show — on the first multi-source SSL corpus of large, feature-rich social networks — that (1) source exposure policy, not corpus size, determines whether small domains survive (with a membership/coverage/synergy/dilution ledger no prior work provides); (2) the objective-combination mechanism that wins on a single graph is not the one that wins for cross-graph transfer, explained by gradient-conflict and forgetting measurements; (3) the two allocations interact, so they must be tuned jointly; and (4) the resulting recipe holds across model scale and adapts cheaply to unseen graphs and tasks.

## Intro
GFMs achieve high performance on a large array of downstream tasks.
No GFM exists for Social Networks because:
Nobody has found additive effect of using multiple (social network) graphs.
No single pretraining objective performs well on all downstream tasks.

### RQs
1. 
## Methodology
### Setup
#### Datasets
| name | ~nodes | ~edges | labels |
|------|-------:|-------:|--------|
| Covid | 23.0M | 107.2M | — |
| Ukraine | 10.4M | 76.9M | — |
| Midterm | 342k | 900k | — |
| Hong Kong | 334k | 1.18M | — |
| TwiBot-20 | 163k | 2.01M | human/bot |
| Election 2020 | 79k | 2.82M | conservative |
| Covid Political | 79k | 181k | conservative |
| Ukraine Suspended | 72k | 354k | suspended |

#### Pretraining Objectives

| Task | Name | Description |
|------|------|--------------|
| NM | Neighbor Matching | Predict which node is a true neighbor of an anchor vs. a sampled non-neighbor. |
| CL | Contrastive | Pull together two augmented views of the same node/subgraph, push apart other nodes. |
| FP | Masked Feature Prediction | Mask a node's input features and reconstruct them from its neighborhood. |
| sLP | Static Link Prediction | Predict whether an edge exists between a pair of nodes. |

*(plus (Dropout) and (Aug) as auxiliary regularizers, not standalone objectives)*

#### Downstream Tasks

| Task | Dataset(s) | Type |
|------|------------|------|
| Bot detection | TwiBot-20 | Node Classification |
| Account suspension | Ukraine Suspended | Node Classification |
| Political leaning | Election 2020, Covid Political | Node Classification |
| Follower count, status count, account age | Covid, Ukraine, Midterm | Node Regression |
| Held-out edge existence | all 8 graphs | Static Link Prediction |
### Experiments
#### Graphs
- 1-graph matrix: Hold NM fixed. Pretrain on each 1-graph, test on all graphs.
- Similarity vs transfer: Measure pairwise divergence of all 8 graphs on topology, feature marginals, and feature–structure coupling. Test whether any divergence axis predicts the measured single-source NM transfer.
- 2-graph sequential vs interleaved: Train sequentially on 2 graphs, test on all tasks.
- 2-graph sampling: Test sampling strictly within-graph and inverse-proportionally between graphs.
- Graph ladder: Train on 1, 2, 3, … graphs, test at each step against NM on all graphs.
#### SSL Objectives
- 1-objective matrix: Pretrain on single graph using one obj. Test performance on a different objective on the same graph.
- Objective ladder: Hold graph fixed. Train on all pairs of objectives, then all 3 (rotation).
#### Learning Anatomy
- Ablation 1: Make all node features the 0 vector, shuffle node feature vectors, try random noise vectors.
- Ablation 2: Add topological info as input features, and try to force the model to use it by changing the objective (structural inputs, LP head, multi-head loss, drop-BN).
## Findings [v0.4]
### Data
- 1-graph matrix. Every graph has a strong specialist. Transfer is asymmetrical. Covid and Ukraine transfer to almost everything; HK donates little and receives little.
- 2-graph pretraining. Naive sequential training results in catastrophic forgetting: train on A->B, and you can only do well on B. Interleaved training fixes this: train on A+B and you do average on both. Better sampling further improves things by allowing you to essentially do as good on both A and B as either single-graph model.
- Multi graph pretraining. 
### Objective
- 1-objective matrix. Transfer only happens when pretext matches task (NM→LP
  is the one cell above floor). On regression, raw features beat every pretrained model.
- Dual objective pretraining. Naive sequential training forgets the same way. But interleaving doesn't fix it like it did for graphs: a pair keeps at most one objective's specialty, and pairs are on average worse than singles.
- Multi objective pretraining. All three at once is different: near-best on classification and regression, and the only model with above-chance link prediction — no single or pair has it. Rewiring edges kills it, permuting features doesn't, so it's real topology. On other corpora the ranking flips though, so this is corpus-specific. Redo with seeds pending.
### Use to motivate things
- Feature ablation 1. NM uses feature content, not topology: noise or zeroed features drop it to chance with edges intact; shuffling features across nodes changes nothing.
- Similarity vs transfer. Feature divergence predicts transfer (ρ ≈ −0.9); topology divergence barely does.
- Feature ablation 2. Forcing topology use failed: the apparent wins were input-scaling artifacts, and the multi-head loss made everything worse. Only the 3-objective rotation produced topology.
## Contribution candidates
- We have a way of leveraging multiple graphs for pretraining **without negative transfer** through interleaved training with better sampling
- We have a way of leveraging multiple SSL objectives
- Possibly, we find that rotation enables emergent skills
- We’re the first to train on an open social GFM at this scale 


## Todo
#### To defend GFM claim, we need to scale:
- [ ] increase model size to 10M+ params
- [ ] train on N=10M nodes
- [ ] train for N steps
#### Other directions we can go into:
- Graph ladder results for sequential training.
- SSL indicator: How well does the SSL performance (or a function of it or its gradient or the activations) indicate downstream performance?
- Scaling: How does downstream performance scale with model and data, especially early on? How does performance evolve as we scale from 1M to 20M+?
- Test the graph ladder models on the downstream tasks.
- Injecting downstream tasks in pretraining for ICL
- Add task: cascade prediction? hashtag prediction?
- Compare another model
- Finetuning: All current experiments are done without finetuning. If we do, we will likely do much better. But allowing this might mean that we should change it everywhere. So maybe avoid. But most other models need fine-tuning.
- Sampling: Uniform sampling over the graph creates clusters and gaps. We have power law degree distributions and currently restrict sampling to nodes with enough edges. Can we do better? Importance sampling? —> again, best to see (a) why we do not reach 100 AUC and 