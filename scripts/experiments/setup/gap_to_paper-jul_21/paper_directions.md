# Stories

## High-level
- What helps/hurts transfer?
- 

## Papers
### Additivity of Data and Task Composition in Graph Foundation Model Pretraining
RQ1 — Data composition: How does the composition of pretraining graphs (single-source, sequential, merged) shape downstream transfer? H1a: interleaving is additive (ladder + 8×8 matrix — you have this). H1b: transfer is asymmetric with donor/recipient structure (you have this). H1c: sequential training suffers forgetting (you have this).

RQ2 — Objective composition: How does the composition of SSL objectives (singles, pairs, triple; rotation schedule) shape downstream ability? H2a: objective mixing is non-additive and interference-prone (you have the lattice: singles, 3 pairs, triple). H2b: the triple produces a 3-way synergy on sLP that no subset shows — this is the headline claim and it currently rests on ~1 seed.

RQ3 — Why the asymmetry? This is what you're missing, and it's what lifts the paper from "we observed" to "we explain." The cheapest strong candidate: gradient conflict analysis — measure cosine similarity of per-objective gradients vs. per-graph gradients during training. The prediction that makes your story cohere: gradients across graphs are near-orthogonal or aligned (hence additive), while gradients across objectives conflict (hence interference). Supplement with representation analysis (e.g., CKA between single-objective and mixed checkpoints) to show what the triple mix builds that pairs don't. These are analysis passes over checkpoints you largely already have.

### What does multi-domain SSL pretraining on social interaction graphs learn, and when does it transfer?
The diagnostic + transfer-matrix paper. RQs: does multi-graph pretraining transfer across events/campaigns; is merging better than specialists; what carries the transfer? Hypotheses (all already supported at 1 seed): H1 — pretraining transfers feature-content structure, not topology (noise/permute ablation); H2 — merging is a robustness trade, not a free win: no OOD bonus, specialists win in-domain, and naive merging is poisoned by a cross-source-discrimination shortcut fixable by within-source sampling; H3 — transfer is predictable from feature-space divergence, not topological divergence, with donor/isolate structure (covid/ukr donors, cp_hk island). Method: your existing 8×8 matrix + ladder + sampling sweeps + ablations, hardened with a seed sweep and the eval-seed fix, plus baselines reviewers now demand (features-only floors — you have them — and a generic GFM zero-shot like GraphAny/OpenGraph). Release the matrix + divergence statistics as the artifact. Highest citability, lowest execution risk — ~80% of the experiments are done.

### Emergent link prediction from SSL objective mixtures
#### The 3-way synergy paper
Hypothesis: capabilities exist that no single objective or pair produces (LLM precedent: UL2/U-PaLM; no graph analog). Needs real new work: the causal edge-rewire ablation proving MIX's LP win is topological, multi-seed replication, budget-matched scale controls, ideally a mixture-weight simplex sweep, and continuous metrics to pre-empt the "emergence is a metric artifact" critique. Highest ceiling (ICLR-shaped), highest risk — it currently rests on one striking but setup-sensitive number.

#### Objective synergy
build the paper around the 3-way SSL synergy. Highest surprise-per-experiment, but currently one result on one model family; making it a paper requires mechanism work (why does the triple unlock link prediction?) and robustness across seeds/datasets, which is where your near-ceiling/seeding issues bite. Better as a section or a workshop paper unless the mechanism story firms up fast.

### The recipe paper
matched-compute comparison showing X% of data/steps suffices, plus what the data must contain. This is the "recipe" idea fused with the ladder. It's attractive because the claim is quotable and practitioner-relevant, and your ladder + trainer infra makes it cheap. But standing alone it's a benchmarking paper; it becomes strong only as the applied payoff section of story A ("because transfer is feature-content-driven and within-source, here's the minimal recipe").

### Anatomy of transfer in social-graph foundation models

RQ1. What property of a source retweet graph predicts its value as pretraining data for a target?

Hypotheses: transfer is governed by (i) feature-space overlap more than topology (your ablation already shows topology-alone ≈ chance), (ii) event/community overlap between sources, (iii) and merging heterogeneous sources creates shortcut incentives that suppress cross-source generalization (your sampling result). Method: you have most of the empirical grid already — what's missing is the predictive layer: compute graph/feature-space distances between sources (embedding distribution divergence, degree/motif profiles, hashtag/actor overlap) and show they predict the transfer matrix. That converts a pile of experiments into one figure and one claim. Risk: low — data is in hand; the intellectual work is the correlational analysis and the writing.

### Anatomy of a social-graph foundation model: what transfers, why, and how to get a generalist. 

RQs:

1. RQ1: What does the workhorse pretext actually learn?
H: SSL on retweet graphs is feature-content matching; transfer is governed by feature-space alignment, not topology. Evidence mostly in hand (ablation + similarity study, two orthogonal methods).
Missing: the interventional single-axis sweep to kill the correlational caveat.
2. RQ2: When does multi-source pretraining beat a specialist?
H: merging is a robustness trade — no OOD bonus, small in-domain tax, big gains only where no strong donor exists — and this is predictable from cheap graph descriptors.
Deliverable: a leave-one-graph-out validated decision rule ("given a new retweet graph, pick donor X / merge / train from scratch"). This converts your 8×8 + ladder from description into a citable transferability-estimation contribution (cluster 2).
3. RQ3: Can one frozen encoder be a generalist?
H: topological capability is emergent from the complete heterogeneous pretext set under rotation, and cannot be constructed by hand-weighted losses. The headline surprise. 
Missing: the causal edge-rewire ablation on MIX, the matched-per-task compute control, and the ParetoGNN positioning.
4. Cross-cutting anchor (your LLM idea, absorbed as a baseline, not the story): "how much does the graph add over the bios at all?" Raw-features/LLM-on-bios floors are already load-bearing in your regression finding — adding an LLM baseline on the labeled tasks defuses the "why GNNs in 2026" review and sharpens the critical narrative.
### Cross-campaign transfer benchmark for social-graph pretraining
The resource paper. Standardized splits, protocols, floors, and the transfer matrix over the 8 graphs (+ optionally the ICWSM 2025 labeled IO campaigns). Cheapest to write, but weaker alone; strongest as A's released artifact or a separate ICWSM dataset paper.

### What are we learning
>Transfer in retweet-graph foundation models is driven by feature-content overlap, not topology; we show which sources help which targets, why merging can hurt, and give a recipe that matches full-data performance with a fraction of the data.

RQs:
- RQ1 (transfer): More sources ≠ better model; gains are in-domain, and we quantify donor structure. (Matrix + ladder — largely done.). Sequential is degenerate; Interleaving is the fix.
- RQ2 (shortcuts): Naive multi-graph merging creates exploitable shortcuts; within-source sampling is the fix. (Done, and it's the prescriptive contribution reviewers want.)
- RQ3 (objectives): Multi-task SSL mixtures show non-additive composition, including an emergent capability. (Done; the paper frames it as evidence that objective mixture matters as much as data mixture.)

### Data-centric transfer study.
> "When does multi-source pretraining help a graph in-context learner? A controlled study on social graphs."

RQs:
- RQ1. Does adding sources help held-out sources (ladder + matrix)?
- RQ2. Does naive graph merging introduce shortcuts (cross-source sampling result)?

Hypotheses mostly already confirmed: gains are in-domain only, covid/ukr are universal donors, merging without within-source sampling is actively harmful. Method: Prodigy pretrained on source subsets, frozen-probe eval + few-shot in-context eval, bot detection (TwiBot-20) as the held-out downstream task. Cites cluster 1 + 4.

### Structure or Semantics? What actually transfers in text-attributed social-graph foundation models.
Features carry node-level transfer; structure is latent and near-useless under single-objective pretraining — except it re-emerges for link prediction, and only when SSL objectives are mixed. Implication: current single-objective GFMs waste the graph, and multi-objective pretraining is how you recover it.

---


### When does multi-source pretraining help on social graphs? An anatomy of transfer in retweet-graph SSL.
**RQ:** when, why, and how should you merge social-graph sources for SSL pretraining?
1. H1 (merging buys membership, not generalization): merged models track column membership — a graph benefits iff it's in training (+.09–.16), pays a small in-domain tax, and merging never helps OOD. (Ladder staircase + 8×8 + OOD null — done.)
2. H2 (the shortcut mechanism + fix): naive merging fails via a source-discrimination shortcut from cross-source negatives — a graph-level instance of contrastive feature suppression — and within-source balanced sampling cures it, rescuing starved small domains above their own specialist (0.31→0.43). (Done, p-sweep bottoms at p=0.)
3. H3 (the explanation): transfer is predicted by feature-cloud divergence, not topology distance, because the dominant pretext is a feature-content learner (noise-vs-permute ablation). (Done correlationally; needs the interventional/matched-family hardening.)

Second act, not headline: only heterogeneous pretext rotation breaks the content-learner ceiling — a 3-way synergy no pair or hand-built multi-head loss reproduces (and the directed3_log overturn strengthens this: with fixed inputs, none of the engineered E1/E2 arms survives, so "composition beats construction" gets cleaner). Framed via Okawa et al.'s multiplicative emergence, defended with a continuous metric, only called "emergent" after engaging Wei and Schaeffer.

## Directions

### Objective-centric.
>"Emergent capabilities from multi-task self-supervised pretraining on graphs."

RQs:
1. Which SSL objectives compose, and do combinations unlock abilities no subset has?
Your 3-way synergy → emergent LP is the headline. Risk: one headline result on one model family, and you don't yet have a mechanism — reviewers will ask why. Cites cluster 2.

> SSL objective composition and emergence.

ParetoGNN established synergy exists; ControlG (Feb 2026) shows interference is unsolved; nobody has mapped the full objective-subset → capability lattice, and your singles .42 → pairs .32 → triple .76 non-monotonicity on link prediction is exactly the shape of an "emergence from objective mixture" result with no graph precedent in print.

---
### Anatomy/shortcut paper.
> "What do graph foundation models learn from social graphs?"

Features vs. topology, shortcut catalog, probing. Interesting but hardest to make constructive; negative-result-heavy papers need a strong prescriptive ending.

> Predicting transfer from graph properties.

Taskonomy for graphs is unclaimed: W2PGNN (KDD'23) predicts pretrain-vs-scratch feasibility, GNNMTE (WWW 2026) does model selection — neither regresses a measured n×n transfer matrix on interpretable graph distances. Your similarity-vs-transfer result (feature-cloud proxy-A-distance ρ ≈ −0.9, degree-KS weakest) is the missing empirical bridge; five distinct audiences (GFM curation, transferability estimation, DA, theory, applied social ML) would cite the matrix as ground truth.

---
### Pretraining Recipe
Naively training on graphs and tasks sequentially does not yield a general model.
[Weihua Hu et al, recipes], and [Maya Bechler-Speicher et al, billion-scale] already found this.
"When does self-supervision help GNNs?” and the newer GFM surveys that all flag "transferability is poorly understood" as the open problem.
But: almost everyone reports that transfer happens; very few papers explain it in terms of the actual graphs.
The single-source 8×8 matrix, the within-source > cross-source shortcut result, and the feature-content ablation are exactly this kind of rare evidence.

> What are the blocks that need to be lifted to allow training at scale? Conversely, what are the enablers of scale?

We already found that:
- interleaved (vs merged) is critical for multi-domain
- sampling is critical under domain imbalance
- 3-way SSL objective allows for emergent abilities in sLP

**But:** We have not found an optimal objective for pretraining that transfers well to all other objectives.

---
### Transfer Learning
An empirical study of transfer learning across domains on the same graph types.
We have a controlled study on domain only.
All other studies investigate transfer across very different graphs, confounding domain/covariate/concept shift and modality shift. 

---
### Shortcut/artifact critiques — what pretraining actually learns.

Hays et al. (WWW'23, bot-dataset artifacts) is a citation magnet; a 2025 critique showed graph-language benchmarks are solvable from a single modality. Your noise-vs-permute ablation (feature content load-bearing, topology ≈ chance) and the cross-source-negatives shortcut are precisely this genre, and novel at the IO/social level — they also directly stress-test the Scientific Reports 2023 claim that graph features dominate in IO detection.

---
### What does multi-source SSL pretraining on social interaction graphs learn, and when does it transfer?
#### RQs:
1. Does multi-graph pretraining transfer across events/campaigns?
2. Is merging better than specialists?
3. What carries the transfer?

#### Hypotheses (_all already supported at 1 seed_):
* H1 — pretraining transfers feature-content structure, not topology (noise/permute ablation);
* H2 — merging is a robustness trade, not a free win: no OOD bonus, specialists win in-domain, and naive merging is poisoned by a cross-source-discrimination shortcut fixable by within-source sampling;
* H3 — transfer is predictable from feature-space divergence, not topological divergence, with donor/isolate structure (covid/ukr donors, cp_hk island).


#### Remaining experiments:
<b>H1</b> — We need to show how well classification (bot, pol, sus) and regression (followers, age, etc) can be solved from just features and just topology:
```
Pretrained GNN
Pretrained GNN, fine-tune 
GNN with just topology
GNN with topology+features
GNN with features
MLP with features
MLP/RF with topology (in-deg, out-deg, centrality?)
```
If the answer is that topology is not really necessary, we need to find another task.
**Or** we skip this and go straight to a task that has very high likelihood of requiring topology (knowing the neighbors is necessary / gives a huge boost)
Maybe add H1.1 — does forcing topology during pretraining help with topology tasks downstream?

---
### Negative Transfer
Negative/conditional transfer and data curation for graph pretraining. Hu et al. 2020's negative-transfer finding is the single most-cited sentence in graph pretraining; APT's "curse of big data" (NeurIPS'23) and [a 2026 graphon-based transfer theory](https://arxiv.org/abs/2605.29828) both call for exactly the corpus-composition evidence you have. There is no "FineWeb for graphs." Well-controlled conditional-transfer results enter an evergreen citation chain: every method paper must cite them as motivation.

---

## Things we could do

1. Why does adding topological information do **worse** than just bios?
> As we go from 0-100% masked features, the model is forced to rely more on topological information (caveat: not linear, should use `n_hop=2`). Should we vary the masking per episode so the model can learn both?

2. Add a downstream task that requires topology (and ideally is a real common CSS task)
3. Why don’t we get 100% accuracy on NM, POL, SUS, BOT? (What are the specific cases)
4. How are we exploring the graph as we sample more? (Mostly supernodes, leaving out important less popular nodes etc)
5. How much of the graph have we explored by episode 40K?
6. Can we outperform the specialist if we fine-tune?
7. Can we reproduce the 3-way SSL result with another model?
8. LLM comparison: further motivates a topology-centric task: if bio features are strong for GTE, Opus will be an unbeatable ceiling.
9. Try a much larger parameter size? But we are already near ceiling for many tasks. The larger param size could be interesting mainly in terms of
10. Add topology features. Then look at the topology weights (and biases) of the net. Then look at the cross-bio-topology weights of the net. Are they mostly 0?
11. Multi-SSL: We have switched objective every episode: what if it switch only every 10 episodes? What if we don’t balance the episodes per objective?
12. When/why does GNN X better than GNN Y?
13. Swap out GraphSAGE for another GNN in PRODIGY (GIN).
14. Another topology-only test with `n_hop=4`, and `limit >> 100`.
15. Compare PRODIGY with a normal GATv2/GraphSAGE encoder.
16. Replicate the emergent sLP result on an OGB graph: Run each objective alone, test on sLP. Then 3-way rotation.
17. Sum up the losses as opposed to rotating them. (Needs to be weighted properly)
18. Scaling experiment: how quickly does transfer saturate? My bet is ~1k episodes is enough.
19. gradient conflict analysis — measure cosine similarity of per-objective gradients vs. per-graph gradients during training. The prediction that makes your story cohere: gradients across graphs are near-orthogonal or aligned (hence additive), while gradients across objectives conflict (hence interference). Supplement with representation analysis (e.g., CKA between single-objective and mixed checkpoints) to show what the triple mix builds that pairs don't.
20. Train PRODIGY from scratch on all the downstream tasks — then we have a baseline to beat.
21. We have only looked at ICL, but we should also consider fine-tuning. Else our models face a harder task than in other papers.
22. Frozen vs. hot embeddings.
23. Can we predict cross-domain transfer for all tasks with NM?



## Conferences
**Social**
```
WWW
ICWSM
```

**ML**
```
AAAI
ICLR
ICML
CMLR
IJCAI
NeurIPS
```