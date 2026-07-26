# What Transfers Across Social Graphs?

## Population coverage and the limits of fixed-budget multi-event pretraining

**Working title:** *Coverage Without Composition? What Transfers in Fixed-Budget Multi-Event Social-Graph Pretraining*

**Status:** independent paper thesis and execution plan, 2026-07-21. This builds substantially on the original [scratch note](./scratch_jul21.md), which remains untouched. Every current headline is one-seed pilot evidence unless explicitly stated otherwise. The point of this document is to decide what deserves replication, not to turn exploratory results into finished claims.

---

## The decision

This should not be a paper about finding the universally “best” pretraining recipe. It should not claim that a 50M-parameter generalist, sequential training failure, an emergent topology capability, or a successful topology head already exists. The repository does not support those stories.

It should also **not** be a source-routing paper. An adversarial audit of the committed 8×8 matrix found that the trivial largest-available-source policy already selects the oracle-best foreign donor for 8/8 targets on NM accuracy and 7/8 on AUC. The proposed feature-distance router is worse. A target-specific selection method is therefore neither needed nor supported by the current domains.

The strongest realistic paper is a mechanistic and evaluation audit:

> **Under a fixed total training budget, what actually transfers when one-hop neighbor matching is pretrained across multiple social graphs: structural knowledge, neighborhood population semantics, exact target membership, or merely the coverage supplied by one broad source?**

The current evidence suggests the following answer:

> **A pooled one-hop NM model consolidates the performance of its strongest constituent source but shows little net gain beyond that source on excluded targets. Large gains occur when the target graph itself enters training. Feature interventions show that NM scoring depends on neighborhood bio features, making feature coverage a candidate explanation for cross-domain transfer—not yet an established transfer mechanism.**

This is not yet the submission claim. The submission-worthy version must show that the result survives training seeds, source-order changes, matched source exposure, event-family and user-disjoint holdouts, a simple bag-of-neighbor-features baseline, and at least a compact downstream test.

### One-sentence paper claim to earn

Under fixed-budget one-hop NM, pooled social-graph pretraining is statistically accounted for by its strongest source; controlled interventions and stronger holdouts identify when this reflects neighborhood population coverage rather than structural or multi-source generalization.

### Why this is more than a negative result

There are three scientifically different outcomes that broad pretraining evaluations often conflate:

1. **Exact membership:** the target graph itself appears in the pretraining curriculum.
2. **Foreign-domain coverage:** a different source already covers the target population well enough to transfer.
3. **Multi-source gain:** several sources together improve beyond every source alone.

A strong foreign donor is genuine generalization. An inclusive all-eight result is not target-excluded generalization. A mixture matching the post-hoc best donor can be useful **generalist consolidation**—one checkpoint matches an oracle chosen from several separately trained checkpoints—but it is not evidence that sources created new capability together.

Separating these phenomena, then identifying the information channel responsible, is the coherent paper.

---

## 1. Problem formulation

Let a graph domain be

\[
G_i=(A_i,X_i),
\]

where \(A_i\) is a social-interaction graph and \(X_i\) contains node bio embeddings. Let \(t\) be a target graph excluded from pretraining, \(S\) a set of foreign sources, and \(U\) a fixed total number of pretraining episodes or optimizer updates.

For a fixed pretext, architecture, sampling policy, and evaluation protocol, define:

- \(T_U(s\rightarrow t)\): transfer after pretraining on source \(s\) for \(U\) updates;
- \(B_U(S,t)=\max_{s\in S}T_U(s\rightarrow t)\): the standard **best-single-source** reference;
- \(M_U(S\rightarrow t)\): transfer from a source-balanced mixture trained for \(U\) total updates;
- \(G_U(S,t)=M_U(S\rightarrow t)-B_U(S,t)\): the **fixed-budget net mixture gain**.

Do not call \(G_U\) “compositional gain.” Under fixed total compute, each source in a \(k\)-source mixture receives roughly \(U/k\) updates while the specialist receives \(U\). Therefore

\[
G_U = \text{multi-source synergy}
- \text{exposure dilution}
- \text{optimization/capacity interference}.
\]

A value near zero does not identify zero synergy; positive synergy could be canceled by dilution. The current result is about the **net value of pooling under a fixed budget**, which is still a meaningful deployment estimand.

### The normalization needed to discuss composition

For representative source sets of size \(k\), compare four conditions:

1. best donor for \(U\) updates;
2. best donor for \(kU\) updates, as an extra-compute/learning-curve control;
3. balanced mixture for \(U\) total updates;
4. balanced mixture for \(kU\) total updates, giving each source about \(U\) expected updates.

Add one larger-capacity mixture at \(U\) for a representative case. Only this design can begin to distinguish insufficient exposure, capacity interference, and genuine lack of synergy.

### Primary research questions

1. **Coverage versus pooling:** Under a fixed total budget, how much net target-excluded gain does a mixture provide over the best single source, and how much of inclusive performance is explained by exact target membership?
2. **Transfer channel:** Does one-hop NM transfer through neighborhood feature distributions, graph structure, or event/user overlap?
3. **Validity beyond the pretext:** Do the source rankings and apparent generalization survive event-family, user-disjoint, temporal, and downstream-task tests?

### Prospectively specified replication hypotheses

These hypotheses were derived after examining the pilot matrix and are **not** pre-registered discoveries. Archive them before launching new runs.

**H1 — fixed-budget consolidation.** Target-excluded \(M_U(S\to t)\) has small net gain relative to \(B_U(S,t)\), while one mixture checkpoint covers several target domains.

**H2 — target inclusion.** Replacing a fraction of foreign-source episodes with target-domain episodes produces a larger target gain than adding more foreign graphs under the same total budget.

**H3 — feature-channel transfer.** Size-normalized coverage of neighborhood bio-feature distributions explains transfer after controlling for event family, language, missing features, and identity overlap better than coarse degree-distribution distance.

**H4 — graph-key holdout is optimistic.** Transfer falls when graph-key exclusion is strengthened to event-family, user-identity, and temporal exclusion.

**H5 — simple-channel falsification.** A fixed DeepSets/bag-of-neighbor-bios baseline reproduces much of the NM transfer geometry. If so, the phenomenon is principally population/text-distribution transfer rather than a general graph-representation result.

Boundary tests—not core hypotheses—ask whether \(n_{hop}=2\), a valid structural pretext, or another backbone changes the active channel and permits positive mixture gain.

---

## 2. What the current evidence actually says

### 2.1 The fixed-budget best-source relationship

The committed 8×8 single-source matrix and cumulative ladder contain 28 target-excluded cells. Seven are the one-source first rung; **21 are true multi-source cells**.

For those 21 multi-source cells, comparing the mixture with its best constituent donor gives the following one-seed pilot values:

- mean absolute net mixture difference: **0.0073 ROC-AUC**;
- mean signed net mixture gain: **−0.0044 ROC-AUC**;
- no cell gains more than 0.010 AUC;
- six cells lose more than 0.010 AUC;
- maximum absolute difference: **0.0201 AUC**;
- pooled Spearman \(\rho=0.977\) and Pearson \(r=0.994\).

The paired differences are the important observation. The pooled correlations are descriptive only: repeated target difficulty, nested source sets, a fixed addition order, and the noisy maximum in \(B_U\) can all inflate them. Hong Kong remains hard and TwiBot-20 remains easy across several rungs, which alone creates strong pooled association.

The current ladder is also dominated by two early broad donors: Ukraine enters first, COVID second, and COVID is the best foreign donor for almost every target. Alternative subsets that exclude those sources are essential.

### 2.2 The “feature router” fails a trivial baseline

On the committed matrix, a largest-available-source policy—COVID except when COVID is the target, then Ukraine—selects:

- the oracle-best foreign donor for **8/8 targets on NM accuracy**, with zero mean regret;
- the oracle-best foreign donor for **7/8 targets on AUC**, with about **0.0004 mean regret**.

The exploratory nearest feature/proxy-A rule selects only 5/8 on AUC, with about 0.0146 mean regret. Thus feature distance is not currently a useful target-specific router. Its possible role is explanatory: perhaps the largest source is also the broadest feature-support source. That must be tested with sample-size-corrected descriptors and an external target, not marketed as a method.

### 2.3 Evidence ledger

| Statement | Current evidence | Safe interpretation | Missing test |
|---|---|---|---|
| Donor transfer is highly directional and heterogeneous. | Complete 8×8 single-source NM matrix; mean off-diagonal donor performance ranges roughly .65–.85 AUC. | Some social graphs are broad donors and others narrow specialists. | Seeds, family-disjoint cells, external target. |
| Fixed-budget mixtures sit near the best-source reference. | 21 true multi-source target-excluded ladder cells; MAE .0073, signed mean −.0044 AUC. | Pooling has little **net** gain under the current 40k budget and order. | Independent subsets, target-centered statistics, exposure-scaled mixtures. |
| Large gains coincide with target inclusion. | COVID-political +.081, Election 2020 +.096, Ukraine-suspended +.165, Hong Kong +.140 when each enters the ladder. | Inclusive breadth partly measures curriculum coverage. | All-seven-foreign model for every target and exposure controls. |
| Specialists beat inclusive all-eight in-domain. | Specialist advantage .006–.039 AUC. | Shared-budget mixtures pay a small in-domain dilution/interference tax. | Seeds, capacity and per-source-exposure controls. |
| Present NM depends on neighborhood feature content. | Intact/permuted feature bags retain NM accuracy; zero/noise collapses across COVID, Midterm, and TwiBot-20. | At one hop, the sampled neighborhood’s feature multiset is load-bearing. | Additional checkpoints and training-time interventions. |
| Feature distance is associated with transfer. | Feature descriptors outperform degree KS in exploratory analyses. | Candidate explanation for the broad-donor pattern. | Full same-family matrix, size correction, language/overlap controls, family holdout. |
| Largest source is already the best foreign donor. | Largest-source policy is oracle on accuracy for 8/8 and AUC for 7/8. | No current target-specific router contribution exists. | External/later event where donor identity is nontrivial. |
| Balanced source-aware sampling helps small sources. | Midterm rises from roughly .31 to .405 at matched compute and .427 at higher compute; specialist is .417. | Balancing mitigates severe underexposure. | 2×2 sampling factorial to separate balancing from within-source confinement. |
| Pretraining improves broad downstream utility. | Regression is currently weak/negative versus raw features; classification lacks all required floors. | Not supported. | Strict target-excluded downstream baseline suite. |

### 2.4 Dataset scope and heterogeneity

The eight source graph artifacts contain about **34.48M graph-node instances** and **191.52M edges** in total. This is not a claim of 34.48M unique people: cross-event identity overlap is unaudited. Two graphs are genuinely massive—COVID at roughly 23M nodes and Ukraine at roughly 10M—while six contain fewer than 350k nodes.

The domains are Ukraine, COVID, Midterm, COVID-political, Election 2020-political, Ukraine-suspended, TwiBot-20, and Hong Kong. They share a social-interaction setting and 768-dimensional bio embeddings, but they are not perfectly uniform retweet graphs: construction pipelines, interaction types/weights, missing-feature behavior, task support, and evaluation query counts vary. The paper must tabulate what is standardized and what is not rather than saying “only the domain changes.”

All names, paths, sizes, capabilities, and provenance should come from [the graph catalog](../../config/graph_catalog.json), using canonical names in prose.

---

## 3. The mechanistic story

### 3.1 What the feature ablation establishes

At \(n_{hop}=1\), the sampled prompt subgraph is essentially a star. Mean/invariant aggregation and absent structural encodings make much rich topology unrepresentable. Edges still decide which feature vectors enter the neighborhood.

Across COVID, Midterm, and TwiBot-20:

- real features work;
- permuting feature-to-node assignments **within the sampled neighborhood** mostly preserves performance;
- zeroing features collapses performance;
- replacing features with distinct but wrong in-distribution noise also collapses performance.

The safe conclusion is:

> The present NM solution depends on the sampled neighborhood’s real feature multiset.

Permutation invariance is partly architectural, not a surprising learned discovery. The result identifies the available information channel; it does not prove that topology is generally useless or that the model had rich topology available and chose to ignore it.

### 3.2 Coverage must be defined on neighborhoods, not raw nodes

If the mechanism is neighborhood-feature content, the explanatory descriptor should reflect neighborhood bags. For each node \(v\), construct a fixed, label-free descriptor

\[
z_v=\phi\left(\{x_u:u\in\mathcal N(v)\}\right),
\]

where \(\phi\) can include the mean, diagonal variance, robust quantiles, or random-feature summaries of the neighborhood’s bio embeddings under a fixed sampling rule. Let \(Z_i=\{z_v:v\in G_i\}\).

A directional coverage measure can then be written as

\[
d_{cov}(t\Vert s)=
\mathbb E_{z\sim Z_t}
\left[\min_{q\in Z_s}\lVert z-q\rVert_2\right].
\]

This is an **explanatory variable**, not a claimed router.

Nearest-neighbor coverage is automatically favored by a source with more samples. Therefore every source must be compared at an equal sampled-neighborhood budget or with a density/extreme-value correction. The pipeline must also:

- fit whitening/dimensionality reduction without transfer-outcome leakage;
- exclude or separately model zero-filled/missing bios;
- deduplicate exact and near-duplicate bios;
- control language, event topic, graph size, and account overlap;
- compare raw-node-feature coverage with neighborhood-feature coverage;
- compare both against topology and feature–structure coupling descriptors.

If corrected neighborhood coverage no longer beats raw source size, the honest result is that data volume—not a new geometry—explains the broad donor.

### 3.3 The proposed causal chain

The paper should test, not assume, this chain:

1. One-hop NM rewards recognition of neighborhood feature distributions.
2. COVID/Ukraine expose the encoder to broad population/language support.
3. Their support transfers to many smaller event graphs.
4. A balanced fixed-budget mixture mostly preserves that broad-donor solution while allocating fewer episodes to it.
5. A niche target improves sharply only when its missing feature regions or collection-specific population enter training.

The current feature ablation plus transfer matrix make this plausible. Training-time feature/topology interventions, lineage controls, and a simple non-GNN baseline are what can make it causal.

### 3.4 Why graph-key holdout may be too weak

COVID and COVID-political, Ukraine and Ukraine-suspended, and multiple U.S. political graphs may share users, event families, collection pipelines, or parent artifacts. Namespacing their nodes inside a merged graph does not remove real-world overlap.

The evaluation hierarchy should therefore be:

1. **graph-key holdout:** target artifact excluded;
2. **event-family holdout:** target and related sibling artifacts excluded;
3. **user-disjoint holdout:** repeated accounts and near-duplicate bios excluded across train/test;
4. **temporal holdout:** a later collection unseen during source choice;
5. **external holdout:** a graph from a different collection pipeline.

Only levels 3–5 can support a strong claim about robust social-graph generalization.

---

## 4. Novelty and citable position

The best-single-source comparison is standard in multi-source domain adaptation, and graph source selection and multi-domain pretraining already exist. The paper must not rename those ideas as new.

The irreducible novelty to earn is **causal identification of the transferable information channel under lineage-aware social-event shift**. The atlas, best-source comparison, and holdout hierarchy are supporting infrastructure. Together they must provide:

1. a controlled, family- and target-excluded transfer audit across heterogeneous-scale social graphs;
2. a precise fixed-budget consolidation result relative to the standard best-source baseline;
3. causal separation of neighborhood population semantics, topology, and overlap;
4. evidence that common graph-key holdouts do or do not survive stronger social-data holdouts;
5. a positive control showing the protocol can detect multi-source benefit when sources are truly complementary.

### Closest work and the required distinction

| Work | What it already covers | What this paper must add |
|---|---|---|
| [PRODIGY: Enabling In-context Learning Over Graphs](https://arxiv.org/abs/2305.12600) | The prompt-graph framework and graph in-context pretraining used here. | A source-domain and information-channel audit in social graphs, not another introduction of graph ICL. |
| [Moment Matching for Multi-Source Domain Adaptation](https://openaccess.thecvf.com/content_ICCV_2019/html/Peng_Moment_Matching_for_Multi-Source_Domain_Adaptation_ICCV_2019_paper.html) (M3SDA) | “Single best” and “source combine” are established multi-source comparisons. | Adapt the comparison to self-supervised graph pretraining, then explain it causally under social-event shift. The max statistic itself is not novel. |
| [GSTBench](https://arxiv.org/abs/2509.06975) | Standardized cross-dataset graph SSL transfer from a large source; several SSL methods fail to transfer. | Directed same-modality source/target analysis, mixtures, family holdouts, and channel interventions. Do not claim the first systematic graph-transfer benchmark. |
| [Better with Less](https://proceedings.neurips.cc/paper_files/paper/2023/hash/b29adb4bf2364acec8fb402ef731bb3b-Abstract-Conference.html) | More pretraining data need not help; active graph-data selection can outperform indiscriminate use. | Source-domain consolidation, matched exposure, and social-population mechanism rather than another generic “more is not always better” claim. |
| [Multi-source Unsupervised Domain Adaptation on Graphs with Transferability Modeling](https://arxiv.org/abs/2406.10425) (SelMAG) | Graph source and node selection using an unlabeled target and learned transferability. | This rules out broad “first graph source selector” claims. Our current largest-source result also removes routing as a contribution. |
| [Graph Data Selection for Domain Adaptation: A Model-Free Approach](https://papers.neurips.cc/paper_files/paper/2025/hash/7b69bc53449ba46bb981951078929a5e-Abstract-Conference.html) (GRADATE) | Model-free graph-data selection using target-side distribution information. | A mechanism/evaluation audit, not a weaker selection heuristic. |
| [All in One and One for All](https://arxiv.org/abs/2402.09834) (GCOPE) and [Text-Free Multi-domain Graph Pre-training](https://arxiv.org/abs/2405.13934) (MDGPT) | Cross-domain graph pretraining, negative transfer, domain alignment, and source interference. | A controlled natural-domain result about what a pooled social-graph model consolidates and which channel causes it. |
| [When to Pre-Train Graph Neural Networks?](https://arxiv.org/abs/2303.16458) (W2PGNN) | Structural feasibility and pretraining-source selection via graphon bases. | Demonstrate when a feature-channel objective is governed by population coverage instead of structure-only similarity. |
| [When Do Graph Foundation Models Transfer? A Data-Centric Theory](https://arxiv.org/abs/2605.29828) and [BRIDGE](https://proceedings.mlr.press/v267/yuan25h.html) | Data discrepancy, curation, bounded multi-domain transfer, and generalization guarantees are already active GFM topics. | A natural social-event test with lineage-aware holdouts and interventions that identify the operative feature/topology/overlap channel. |
| [Simplistic Collection and Labeling Practices Limit the Utility of Benchmark Datasets for Twitter Bot Detection](https://arxiv.org/abs/2301.07015) | Simple dataset artifacts can explain high within-dataset bot performance and weak cross-dataset generalization. | Extend the audit from task labels to graph pretraining itself, with explicit user, family, feature, and topology controls. |

The strongest novelty sentence is:

> We apply the standard best-source stress test to fixed-budget self-supervised social-graph pretraining and use controlled interventions and stronger holdouts to determine whether apparent cross-graph generalization is structural or population-semantic.

---

## 5. Experiments, ordered by decision value

### E0 — Correctness, lineage, and no-training falsification gate

Before GPU work:

1. Check in a script that reconstructs all 21 multi-source target-excluded cells, their source sets, \(B_U\), \(M_U\), and paired differences.
2. Reproduce the largest-source baseline for both accuracy and AUC in that script.
3. Build a dataset-lineage table: parent collection, event, time window, interaction type, feature pipeline/version, labels, and graph-construction script.
4. Audit exact account overlap, exact and near-duplicate bios, language distributions, missing/zero features, and temporal overlap.
5. Define graph-key, event-family, user-disjoint, and temporal splits before new training.
6. Archive the replication hypotheses, primary metrics, checkpoints, and source-set cells.
7. Quarantine the current static-link result. Its evaluator does not condition the score on both queried endpoints, and score polarity is not validly calibrated. It cannot establish learned topology or three-objective emergence.

**Gate:** if identity or lineage overlap explains the broad-donor pattern, reframe the paper around leakage/population reuse before training more mixtures.

### E1 — Seed-harden the directed transfer atlas

Train the eight single-source specialists to three total independent seeds first. Reusing the existing seed after exact config verification requires **16 new pretraining runs**.

Use the E1 variance to perform a prospective power/equivalence calculation. If a one-AUC-point fixed-budget equivalence claim needs five seeds, add them; do not assume three seeds are sufficient.

Evaluate all sources on all targets with paired episode sets and report:

- per-target NM accuracy as the primary transfer metric;
- AUC as secondary because it is near saturation in some cells;
- training-seed and evaluation-episode uncertainty separately;
- directional asymmetry \(T(s\to t)-T(t\to s)\);
- donor rankings after family/user-overlap controls;
- DeepSets/bag-of-neighbor-bios, raw-feature, and topology-only reference scores where feasible.

The best donor must be chosen from seed-mean or hierarchical latent performance and evaluated on independent episode splits to reduce winner’s-curse bias.

### E2 — Target inclusion versus foreign coverage

For every target \(t\), train an **all-seven-foreign** model. Compare:

1. target specialist;
2. inclusive all-eight mixture;
3. all-seven-foreign mixture;
4. best foreign specialist;
5. largest foreign specialist.

Call all8−all7 the **curriculum-inclusion contrast**, not a perfectly isolated membership effect: all8 and all7 allocate different fractions of the budget to each foreign source. For a representative subset, add an all-seven checkpoint with foreign-source exposure matched to all8.

Stage this experiment:

- first run one seed for all eight target exclusions as a go/no-go check;
- only if the pattern survives, complete three seeds.

The full three-seed package is about **23 new runs**, because only one current Hong-Kong-excluded seed may be reusable after verification. Two additional inclusive all-eight seeds are also required for a three-seed comparison.

### E3 — Exposure and capacity normalization

For one representative source set, run the four \(U\)/\(kU\) conditions defined in Section 1 and add one larger-capacity fixed-\(U\) mixture. Reuse E1/E4 checkpoints only when source set, sampler, budget, architecture, and seed match exactly. A second source set is a conditional replication after the first identifies whether exposure or capacity matters.

This is essential to interpret the null:

- if \(M_U\approx B_U\) but \(M_{kU}>B_{kU}\), composition exists but was hidden by dilution;
- if longer donor training explains the same gain, extra optimization—not diversity—was responsible;
- if a larger model helps only the mixture, capacity interference matters;
- if none helps, lack of useful complementarity becomes more plausible.

Report actual optimizer steps, sampled nodes/edges, wall time, peak memory, and FLOPs estimates. Under the present matched-40k protocol, a single-source and all-seven model do **not** differ materially in training steps; source count alone is not compute savings.

### E4 — Break the fixed order and add a positive control

The cumulative ladder starts with the two broad donors, so it cannot identify general source-set behavior. Predeclare four to six global source mixtures spanning:

- with and without COVID;
- with and without Ukraine;
- two individually weak but putatively complementary sources;
- redundant same-family sources;
- source-set sizes 2, 4, and 6 where target exclusion permits.

List every intended \((S,t)\) cell before training; a global mixture supplies multiple held-out targets, but not every target is eligible for every set.

Add a **positive-control composition task** by partitioning one graph or synthetic neighborhood-feature support into two source domains whose union matches a target while neither source does alone. Normalize gain by headroom to the target specialist. If the protocol cannot detect gain in this constructed complementary case, a real-data null is uninterpretable.

Four to six real-data source-set configs at three seeds imply **12–18 new runs**. The positive control additionally needs source-A, source-B, and source-A∪B models at three seeds—about **9 runs** unless exact checkpoints already exist. E4 therefore costs roughly **21–27 new runs**, not an open-ended subset sweep.

### E5 — Test the active information channel

Use a compact, predeclared mechanism package on two broad donors and two difficult targets:

1. **Bag baseline:** a fixed DeepSets/permutation-invariant model over neighborhood bios, without graph message passing beyond neighbor selection.
2. **Feature support:** reweight or swap source feature distributions while preserving adjacency and episode counts.
3. **Topology:** degree-preserving and, if feasible, community-preserving rewiring while preserving features.
4. **Binding:** retain neighborhood feature multisets but disrupt feature-to-node assignment.
5. **Missingness/duplicates:** repeat after removing zero features and duplicate bios.

The decisive comparison is neighborhood-feature coverage versus marginal-node coverage, graph size, language, overlap, and topology descriptors under target-centered analysis.

Evaluation-time interventions are an initial screen. At least one feature and one topology intervention must be repeated during pretraining for a causal claim.

### E6 — Stronger holdouts and downstream go/no-go

The broad paper requires evidence beyond NM→NM. Start compactly with two contrasting classification targets: one where raw bios are highly predictive and one lower-homophily/less feature-predictable target. Exclude the target, its event-family siblings, and overlapping accounts from pretraining.

Baselines:

- raw GTE bio logistic regression/MLP;
- DeepSets/bag-of-neighbor-bios;
- topology-only and degree controls;
- random frozen encoder;
- from-scratch supervised GNN;
- frozen and fine-tuned foreign specialist;
- frozen and fine-tuned all-seven-foreign mixture as the graph-key-holdout baseline;
- frozen and fine-tuned **family-excluded** mixture, with a separately reported user-filtered version.

Predeclare label budgets, transductive versus inductive access, fine-tuning seeds, checkpoint selection, and whether degree features use full-graph information.

The regression panel contains **up to six profile targets across four graph domains**; TwiBot-20 lacks at least one target. Treat regression as a secondary negative/control axis unless the corrected protocol changes the present result that raw features win.

If source rankings and interventions do not predict downstream utility, scope the paper explicitly to NM transfer. A broad graph-pretraining claim is then a no-go.

### E7 — Sampling factorial, appendix/gated follow-up

Current sampling experiments partially conflate source balancing and within-source episode confinement. A 2×2 factorial can separate them:

| Episode construction | Source probability |
|---|---|
| cross-source allowed | proportional |
| within-source only | proportional |
| cross-source allowed | balanced |
| within-source only | balanced |

This is an experimental-control appendix, not a headline. Until run across seeds, say balanced source-aware sampling **mitigates severe underexposure**; do not claim a shortcut was causally removed.

### E8 — Flagship extension

For a broad ML claim, add:

- \(n_{hop}=2\) or a second backbone/depth;
- a feature-reconstruction objective;
- a correctly pair-conditioned structural objective and evaluator;
- an external or temporally later graph;
- enough target domains to distinguish domain-level effects from eight correlated artifacts.

Then ask whether the active transfer geometry changes with the pretext channel. This is the route to the larger thesis “transfer follows the pretext.” It is not existing evidence.

### Staged run and storage budget

| Stage | New pretraining runs | Decision |
|---|---:|---|
| E0 | 0 | Stop/reframe if overlap or the largest-source baseline explains the story. |
| E1 specialists to 3 seeds | 16 | Estimate training variance and donor stability. |
| E2 first-pass all-seven-foreign | 8 | Go/no-go before full replication. |
| E2 completion + inclusive seeds | roughly 17 | Complete the target-inclusion decomposition. |
| E3 exposure/capacity controls | roughly 9–12 | Determine whether fixed-budget dilution explains the null on one representative source set. |
| E4 real mixtures + positive control | roughly 21–27 | Test order independence and verify that the protocol can detect complementarity. |
| E5 causal core | roughly 6–12 | Establish the active channel. |

The **E0–E5 core alone** is therefore roughly **77–92 new pretraining runs** at three seeds. E6 is intentionally unpriced until E0 reveals which family- and user-filtered artifacts can reuse checkpoints; strict holdouts generally require new artifacts and pretraining as well as cheap probe/fine-tuning runs. E7 and E8 are separate gates. A power analysis that requires five seeds would increase the budget further and must be costed before launch. This is a staged program, not a promise to launch every row at once.

Storage is a first-class constraint. Existing merged graph artifacts are around 100GB; eight leave-one-out plus subset artifacts could approach 1–1.5TB. Prefer a loader that samples from separate source artifacts without materializing every union. If a new graph artifact is required, update [the graph catalog](../../config/graph_catalog.json) first and include construction/storage time in the plan.

---

## 6. Statistical design

The domain-level sample is eight correlated graph artifacts, not 64 independent observations. More evaluation episodes reduce measurement noise but do not create more domains.

Required analysis:

- lead with per-target paired \(G_U(S,t)\), not pooled correlation;
- target-center transfer values and report within-target slopes;
- compare against target-ID, source-count, rung-order, and largest-source baselines;
- use hierarchical source/target effects or multiway-clustered uncertainty;
- use matrix/QAP-style permutation tests for descriptor–transfer association;
- preserve paired evaluation episodes across checkpoint comparisons;
- separate training-seed variance from episode variance;
- correct the noisy maximum in \(B_U\) by cross-fitting donor selection and evaluation or modeling latent seed-mean performance;
- use equivalence tests only after a prospective power analysis and justified practical margin;
- normalize gains by available headroom where target metrics approach saturation;
- report each target and event family, not only a grand mean.

Any target-conditioned descriptor analysis has at most eight correlated targets. Do not fit flexible metric weights. Use predeclared scalar descriptors, and require an untouched external target before reviving a prospective selection claim.

---

## 7. The narrative in paper order

### Act I — Broad evaluation can confuse membership with transfer

Show the inclusive all-eight result and the ladder jumps. Introduce exact membership, foreign coverage, and multi-source gain as separate concepts.

### Act II — One broad source accounts for fixed-budget OOD performance

Show the seed-hardened directed atlas, largest-source baseline, and paired net mixture differences. Present mixture≈best-source as consolidation under a fixed budget, not as proof that information can never compose.

### Act III — Is the cross-domain channel neighborhood population semantics?

Use the bag baseline, feature/topology interventions, and size-corrected neighborhood coverage. Determine whether COVID/Ukraine transfer because they cover languages/populations, because of overlap, or because of actual structural knowledge.

### Act IV — Stronger holdouts reveal what “unseen” means

Compare graph-key, event-family, user-disjoint, temporal, and external holdouts. This is where the social-data contribution becomes more than another graph SSL matrix.

### Act V — Test downstream relevance and the boundary

Use compact classification experiments and raw/bag/topology controls. If the effect is NM-specific, say so. If a deeper or structural objective changes the transfer geometry, use it as the boundary result rather than retrofitting a universal claim.

---

## 8. Claims and contributions

### Claims that could be defensible after the required experiments

1. Under a specified fixed total budget, pooled one-hop NM has little net target-excluded gain over the standard best-single-source reference.
2. A pooled checkpoint can consolidate broad-donor performance across targets, but large gains on niche domains mostly reflect direct curriculum inclusion.
3. The transferable signal in the present setup is primarily neighborhood bio-feature/population coverage, conditional on overlap and language controls, rather than rich graph structure.
4. Graph-key holdout overstates or accurately reflects generalization by a measured amount once event-family, identity, and temporal exclusions are imposed.
5. The conclusions transfer to downstream tasks—or are explicitly scoped to the NM pretext if they do not.

### Contribution set

1. A lineage-aware, family- and target-excluded transfer audit for eight heterogeneous-scale social graph artifacts.
2. A robust empirical result about fixed-budget pooled NM relative to the **standard** best-source baseline, with exposure and positive controls.
3. Causal evidence separating neighborhood population semantics, topology, and overlap as transfer channels.
4. A measurement of how conclusions change across graph-key, event-family, user-disjoint, temporal, and external holdouts; the hierarchy itself is not novel.
5. Releasable transfer matrices, descriptors, splits, configs, and audit artifacts.

The best-source maximum, source selection, and model weights are not standalone scientific contributions.

---

## 9. What is cut or quarantined from the original scratch

### “Find the best way to pretrain”

Cut. “Best” is undefined over objectives, architectures, budgets, domains, and tasks. The replacement is a fixed-budget transfer audit with explicit normalization.

### “An open 50M-parameter model”

Cut. The default NM model is approximately **1.64M parameters**, not 50M. Model scale is not the contribution.

### “Sequential pretraining is bad”

Cut. Existing sequence effects are roughly −0.56 to −0.87 percentage points, confidence intervals cross zero, and adjusted \(p=1.0\).

### “Interleaving/PretrainStrats makes it best”

Demote. Balanced source-aware interleaving is a fairness/control choice. Its causal mechanism needs the E7 factorial.

### “90% of performance in 10% of the steps”

Cut. No complete matched scaling study supports it.

### “A small topology head gives a huge boost”

Quarantine. The current static-link evaluator is not truly pair-conditioned; the apparent MIX emergence result is invalid as topology evidence.

### “Follower count is a topology-requiring task”

Cut. Follower count is a profile attribute, not retweet-graph degree, and may correlate with profile semantics.

### “NM had topology and chose not to use it”

Rewrite. The one-hop invariant architecture cannot represent much topology. The evidence shows dependence on a neighborhood feature multiset.

### “Feature distance gives a practical router”

Cut. Largest source is already oracle-best for the current accuracy matrix and nearly oracle on AUC. Feature coverage remains a mechanism hypothesis only.

### “Best-donor envelope is a new diagnostic”

Cut as a novelty claim. Best-single-source versus source-combine is standard in multi-source domain adaptation. The contribution must be the controlled graph-pretraining application plus causal explanation and stronger holdouts.

### “The same result holds across multiple models”

Cut until run. A second depth/backbone is an extension.

### “Four classification and four regression tasks prove generality”

Rewrite. Classification still needs strict foreign-pretraining and raw/random/bag/scratch floors. Regression is presently negative and contains up to six targets across four domains, not a won four-task result.

---

## 10. Figures and tables

Four main figures are enough:

1. **Directed transfer atlas + largest-source reference.** Seed means, asymmetry, event-family annotations, and overlap markers.
2. **Fixed-budget consolidation.** Per-target mixture−best-source differences, all8/all7/best/largest decomposition, and exposure-scaled controls.
3. **What channel transfers?** Bag baseline, feature/topology interventions, and target-centered descriptor associations.
4. **How holdout strength changes the conclusion.** Graph-key → family → user-disjoint → temporal/external, with compact downstream results.

Main tables:

1. Dataset lineage, event family, time, language, interaction type, scale, missingness, identity overlap, and supported tasks.
2. Net mixture-gain equivalence/uncertainty by target and budget normalization.
3. Raw-feature, bag, topology-only, random, specialist, and pooled downstream baselines.

Put the full eight-rung ladder, all source-set cells, sampling factorial, and descriptor catalog in the appendix.

---

## 11. Draft abstract to aim for

> Broad graph pretraining is commonly evaluated on datasets represented in, or closely related to, the pretraining curriculum, conflating target membership, foreign-domain coverage, and genuine multi-source gain. We disentangle these effects for one-hop neighbor-matching pretraining across eight heterogeneous-scale social graph artifacts. Under a fixed total training budget, pooled target-excluded performance shows [effect and interval] net gain over the standard best-single-source reference, while direct target inclusion produces [effect and interval]. Because fixed-budget pooling also dilutes per-source exposure, we repeat the comparison under exposure- and capacity-normalized controls and a complementary-source positive control. Feature, topology, identity-overlap, and bag-of-neighborhood-bios interventions show that [mechanism result]. Strengthening evaluation from graph-key to event-family, user-disjoint, and temporal/external holdouts changes transfer by [result], and compact downstream experiments establish [scope]. These findings distinguish consolidation from composition and show what information actually transfers in multi-event social-graph pretraining.

Do not put the current one-seed correlations in the submitted abstract. The primary result should be paired, target-centered net gain with seed intervals.

---

## 12. Paper outline

1. **Introduction**
   - Why broad curriculum performance is ambiguous.
   - Membership, foreign coverage, consolidation, and mixture gain.
   - Main result and social-data implication.
2. **Setting and audit protocol**
   - Graph lineage, model/pretext, budgets, holdout hierarchy, metrics.
3. **A directed atlas of social-graph transfer**
   - Donors, recipients, asymmetry, largest-source result.
4. **Does fixed-budget pooling add value?**
   - Best-source baseline, all-seven-foreign models, exposure/capacity controls, positive control.
5. **What information channel transfers?**
   - Neighborhood bags, features, topology, overlap, size-normalized coverage.
6. **Does graph-key generalization survive stronger holdouts?**
   - Family, identity, temporal/external, downstream tests.
7. **Related work**
   - Graph SSL transfer, multi-source adaptation, multi-domain GFMs, social-data artifacts.
8. **Discussion and limits**
   - One-hop architecture, bio features, correlated domains, release constraints.

---

## 13. Falsification and decision rules

- **If user/event-family overlap explains donor transfer**, reframe around population reuse and evaluation leakage; do not claim representation geometry.
- **If target-centered, multi-seed net mixture gains are positive**, replace “coverage without composition” with a conditional-composition paper and identify which source sets create it.
- **If exposure-scaled mixtures improve but fixed-budget mixtures do not**, conclude that composition is compute-limited, not absent.
- **If the positive control cannot produce gain**, the test lacks sensitivity; a real-data null is not publishable as no composition.
- **If graph size alone explains the broad donor after sample correction**, drop the neighborhood-coverage novelty and write the result as a scaling/data-volume effect.
- **If DeepSets matches the GNN transfer matrix**, a paper remains significant only if stronger user/event/time exclusions reveal a sharp, quantified consequence or another result beyond the existing dataset-artifact literature. Otherwise the story is too incremental; avoid a broad structural-GFM claim.
- **If topology interventions dominate features after retraining**, revise the active-channel story.
- **If event-family/user-disjoint holdouts preserve transfer**, that is stronger evidence than the current graph-key result and should become the headline.
- **If NM rankings do not predict downstream utility**, scope to NM or stop the broad paper.
- **If seed variance is comparable to the observed ~.007 differences**, report consolidation with wide equivalence bounds rather than ranking mixtures.

### Go/no-go order

1. E0: lineage, overlap, reproducible paired analysis, largest-source baseline.
2. E1–E2 first pass: seeds and all-seven exclusions.
3. E3: determine whether the fixed-budget result is dilution.
4. E4: verify the result across source composition and validate test sensitivity.
5. E5: earn the mechanism claim.
6. E6: earn the broad graph-pretraining claim.
7. E7/E8 only after the core survives.

---

## 14. Venue and significance

The mechanism-plus-holdout version is naturally a WWW or ICWSM paper: it studies how social-event graph models transfer, how collection/population overlap affects evaluation, and whether “graph generalization” is actually neighborhood text generalization. A graph/data-mining venue is plausible if the exposure-normalized mixture result and positive control are strong.

An ICLR/NeurIPS-style claim requires the E8 boundary package: another pretext/channel, another depth/backbone, and an external target. Eight correlated social artifacts alone cannot support a universal GFM principle.

The reusable contribution is not a new max statistic. It is a disciplined audit: compare to best source, normalize exposure, identify the active channel, and strengthen the holdout until population reuse cannot explain transfer.

---

## 15. Existing evidence map

- [Consolidated findings](../../scripts/experiments/analysis/consolidate_7_20/FINDINGS.md)
- [8×8 single-source NM matrix](../../scripts/experiments/analysis/nm_single_source_matrix/FINDINGS.md)
- [Eight-rung mixture ladder](../../scripts/experiments/analysis/nm_ladder/RESULTS.md)
- [Matched merged-versus-single study](../../scripts/experiments/analysis/nm_transfer_matrix/RESULTS.md)
- [Cross-source sampling study](../../scripts/experiments/analysis/nm_cross_source_shortcut/RESULTS.md)
- [COVID/Midterm imbalance study](../../scripts/experiments/analysis/nm_covid_midterm/RESULTS.md)
- [Feature ablation](../../scripts/experiments/analysis/feature_ablation/FINDINGS.md)
- [Similarity-versus-transfer pilot](../../scripts/experiments/analysis/similarity_vs_transfer/FINDINGS.md)
- [Frozen-probe matrix](../../scripts/experiments/analysis/pretrain_probe_matrix/FINDINGS.md)
- [Graph catalog](../../config/graph_catalog.json)

The static-link, multi-objective rotation, objective-pair, and topology-feature analyses are intentionally excluded from the positive evidence map. They may guide evaluator repair, but they do not currently support topology or emergence claims.

---

## Bottom line

The paper is not “we tried many pretraining strategies and found the best one,” and it is not “feature distance picks the right source.” It is:

> **A fixed-budget pooled social-graph encoder can look broadly transferable while mostly consolidating one large donor and directly covering known targets. The scientifically important question is which information channel survives truly foreign event, user, and time shifts.**

The existing matrix, ladder, and feature ablation make that question concrete. Exposure controls, lineage-aware holdouts, a bag-of-neighbor-bios baseline, and compact downstream validation are the bounded route from an interesting one-seed pattern to a citable and honest paper.
