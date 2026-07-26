<!-- TODO
is the goal for downstream tasks:
good 0-shot performance
minimize need for FT
get best performance after FT -->


# Title
## Abstract
We introduce SocialGFM (PRODIGY with our recipe) and SocialGraphBench (though we cannot release that).
We train SocialGFM on G graphs from D domains using O pre-training objectives.
We demonstrate that training PRODIGY with our recipe beats more recent models despite fewer parameters.
In the few-shot setting, we demonstrate that SocialGFM reaches comparable performance to specialist GNNs on unseen graphs, and outperforms specialists when fine-tuned.
In the zero-shot setting, we demonstrate that SocialGFM’s representations beat comparable models with fine-tuning.
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
## Results
See folders.
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
## Contributions
- Model
- Recipe
  - how to generalize to multiple domains (We have a way of leveraging multiple graphs for pretraining **without negative transfer** through interleaved training with better sampling)
  - how to generalize to multiple tasks ()
  - how to scale to 100M data points (if naive scaling does not work)
  - how to scale to 20M parameters (if naive scaling does not work)

#### Contribution candidates
- Non-separability — maybe data and objective interact, currently untested.
- Mechanism — What actually transfers between domains and tasks.

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
- Joint graphs over disjoint graphs.