# Which Graphs for Which Objectives?

## Channel-aware joint mixtures for transferable social-graph pretraining

**Working subtitle:** *The unit of graph pretraining is a graph–objective pair.*

**Status:** v2 paper thesis and execution plan, 2026-07-22. This develops the direction in the original [scratch note](./scratch_jul21.md) without modifying it or treating its provisional claims as constraints. The adjacent [coverage-focused v1](./v1_coverage_jul21.md) remains a useful account of the current neighbor-matching evidence; v2 is the more ambitious paper to attempt.

Current-result labels used throughout:

- **[EXISTING — 1 seed]**: measured in the repository, but not yet a submission-grade estimate;
- **[EXPLORATORY]**: useful for choosing experiments, not for a headline;
- **[OUTSTANDING]**: required new evidence, with a result slot left deliberately blank;
- **[QUARANTINED]**: the present evaluator does not support the intended claim.

---

## The decision

Keep the original direction—multi-graph pretraining, interleaving, sampling, multiple self-supervised objectives, and transfer to unseen social graphs—but change the scientific unit of analysis.

The paper should not ask which graph is universally best, whether more graphs are always better, or whether one pretext is generally best. It should ask:

> **Which graph should supply which self-supervised objective, and why?**

The proposed answer is:

> **A pretraining objective exposes a particular information channel in a graph. That channel induces its own cross-graph transfer geometry, so the value of a source cannot be separated from the objective applied to it. A good fixed-compute generalist therefore allocates episodes jointly over graph–objective pairs, rather than choosing a data mixture and an objective mixture independently.**

This produces one coherent progression:

1. causally identify what each objective can learn from features, structure, and their coupling;
2. show that different objectives induce different source-to-target transfer geometries;
3. demonstrate that source and objective allocation is non-separable;
4. use those measurements to construct a robust joint mixture;
5. test whether the mixture transfers across held-out event families, tasks, scale, and one truly external graph.

The existing neighbor-matching results motivate this thesis: one-hop NM behaves like a neighborhood-feature-bag objective, broad source graphs transfer widely, small graphs are badly underexposed by naive sampling, and exact target membership matters. Those results are one channel-specific slice of the proposed paper—not the whole paper.

### One-sentence claim to earn

Across heterogeneous social graphs, self-supervised objectives transfer along distinct, causally identified information channels; modeling this objective-conditioned geometry and allocating a fixed compute budget non-separably over graph–objective pairs improves both average and worst-family held-out transfer over sequential, proportional, uniform, objective-only, data-only, and factorized mixtures.

### The hard novelty test

The paper is not novel merely because it mixes several datasets and several losses. Graph data selection, multi-objective graph SSL, objective scheduling, and generic joint source/task weighting already exist. The contribution survives only if all three statements are supported:

1. **Channel crossover:** interventions establish that at least two objectives rely on meaningfully different information channels.
2. **Objective-conditioned geometry:** an objective-specific distance predicts held-out transfer better than a single universal graph distance.
3. **Non-separable value:** a channel-derived graph–objective allocation outperforms rank-1/factorized and rank-2 counterparts with the same information and tuning budget under matched compute, while shuffled-channel controls do not.

If these fail, do not relabel a generic multi-task scheduler as the contribution.

---

## 1. Problem formulation

Let source graph `s` be

\[
G_s=(A_s,X_s),
\]

where `A_s` is the interaction structure and `X_s` contains node bio embeddings. Let

- `s ∈ S` index source graphs;
- `o ∈ O` index self-supervised objectives;
- `t ∈ T` index held-out target graphs;
- `q ∈ Q` index evaluation tasks;
- `B` be a fixed encoder-compute budget, measured primarily in encoded node/edge tokens and checked in FLOPs;
- `π_{s,o}` be the fraction of `B` allocated to graph–objective cell `(s,o)`.

The allocation satisfies

\[
\pi_{s,o}\geq 0,\qquad \sum_{s,o}\pi_{s,o}=1.
\]

Objective-specific episode rates implement this compute allocation; update counts are reported but are not assumed to have equal cost. A conventional pipeline makes two independent choices,

\[
\pi_{s,o}=p_s q_o,
\]

where `p_s` is a source mixture and `q_o` is an objective mixture. The central hypothesis is that this factorization is generally wrong: masked feature denoising may be useful on one source while pairwise structure reconstruction is useful on another.

### Information channels and their estimand

Operationalize three non-overlapping factors on the synthetic test bed before examining the main transfer outcomes:

1. **feature channel `X`:** the marginal node-feature distribution `P(X)`, without reference to adjacency;
2. **structure channel `A`:** the unlabeled topology distribution `P(A)`;
3. **coupling channel `XA`:** residual dependence between features and structural positions after fixing both marginals.

Synthetic factorial graphs vary one factor while holding the other two fixed and verify those invariances numerically. Natural graph perturbations cannot generally do this—feature resampling and rewiring often change coupling too—so they are external-validity stress tests, not the source of orthogonal causal weights.

For objective `o`, seed `r`, synthetic graph family `g`, and channel `c`, define a paired training-time effect

\[
\Delta_{o,c,g,r}=
\frac{M_o(\theta^{base}_{o,g,r};G^{clean}_g)-M_o(\theta^{-c}_{o,g,r};G^{clean}_g)}
{M_o(\theta^{base}_{o,g,r};G^{clean}_g)-M_o(\theta^{random}_{g,r};G^{clean}_g)+\epsilon},
\]

where `θ^{-c}` is trained at matched compute with channel `c` removed, then every model is evaluated on the same clean held-out synthetic graph using pretext-specific diagnostics and planted channel probes. No cross-graph transfer or downstream outcome enters this estimate. Preserve signed effects; form the nonnegative fingerprint `a_o` from the positive hierarchical mean effects only when their sum exceeds a preregistered random-variation threshold. Otherwise declare the fingerprint undefined rather than forcing it onto a simplex.

Lock the checkpoint at the common compute endpoint. The primary clean metrics are NM accuracy, MFD explained variance over masked coordinates relative to the training-fold mean, and validation-oriented Pair-LP ROC-AUC; planted feature-, structure-, and coupling-probe scores are reported separately rather than averaged opportunistically. Channel removal changes pretraining context only—the sampled query identities, supervision targets, and clean evaluation distribution are held fixed.

Natural evaluation-time corruption answers the weaker question “what does this trained model rely on under this stress?” It does not by itself identify what pretraining causally taught.

For directional source coverage, let `d_c(t || s)` be the equal-sample target-to-source discrepancy in channel `c`. Define two different outcomes and do not conflate them:

- `Y_pre(s,o,t)`: clean transfer on objective-near evaluation for objective `o`;
- `Y_down(s,o,t,q)`: transfer to downstream task `q`.

The objective-conditioned pretext discrepancy is

\[
D_o(s\rightarrow t)=\sum_c a_{o,c}d_c(t\Vert s).
\]

For downstream task `q`, estimate a nonnegative task-demand fingerprint `b_q` from labeled development tasks using channel ablations and mandatory raw/bag/structure/random/scratch floors. Recompute it inside every outer fold. Define overlap and a task-conditioned discrepancy as

\[
\kappa_{o,q}=\sum_c a_{o,c}b_{q,c},\qquad
w_{o,q,c}=\frac{a_{o,c}b_{q,c}}{\kappa_{o,q}+\epsilon},
\]

\[
D_{o,q}(s\rightarrow t)=\sum_c w_{o,q,c}d_c(t\Vert s).
\]

This separates “the source resembles the target for objective `o`” from “objective `o` teaches channels that task `q` needs.” Validate the product rule for `w` on development folds; if it is not predictive, do not use it in ChannelMix.

### Two deployment settings

Keep the primary and secondary settings separate.

1. **Target-domain-agnostic, task-aware generalist — primary.** The downstream task panel is declared and its development labels are available, but the final target domain, its descriptors, and its labels do not influence the mixture. Learn one joint policy using nested pseudo-target event-family folds.
2. **Unlabeled-target, known-task mixture — secondary/transductive.** If the unlabeled target graph is available, compute its descriptors for the same known task panel but never use target labels. Compare with target-aware factorized and nearest/coverage baselines using the identical descriptors. Make no unseen-label-function claim.

The primary paper is therefore not dependent on a fragile source router; the external graph is evaluated fully inductively first.

---

## 2. Research questions

### RQ1 — Channel identification

**Which information channels do the pretraining objectives actually use?**

Do feature, structure, and feature–structure-coupling interventions cause stable, objective-specific degradation patterns across graphs and seeds?

### RQ2 — Objective-conditioned transfer geometry

**Does each objective induce a different map of which graphs transfer to which others?**

Does `D_o(s→t)` predict objective-near transfer, and does the task-conditioned `D_{o,q}(s→t)` predict downstream transfer, in leave-event-family-out evaluation better than graph size, event identity, a universal graph distance, or a learned outcome-only baseline?

### RQ3 — Non-separability

**Can graph choice and objective choice be made independently?**

Is there a reproducible source × objective interaction, and does a full joint allocation `π_{s,o}` outperform the best factorized allocation `p_s q_o` at the same encoded node/edge-token budget and matched encoder FLOPs?

### RQ4 — Generalist pretraining

**Can a channel-aware joint mixture improve a single model’s average and worst-family transfer?**

Compare against source specialists, objective specialists, sequential blocks, size-proportional interleaving, uniform joint mixing, source-only selection, objective-only scheduling, source-balanced factorized mixing, and a generic joint reweighting baseline.

### RQ5 — Generalization and scale

**Do the channel fingerprints and learned allocation survive changes in event family, downstream task, backbone, training duration, and model size?**

The decisive tests are an untouched later/external social graph, a second message-passing backbone, and at least one feature-heavy and one genuinely topology-required downstream task.

---

## 3. Prospectively specified hypotheses

These hypotheses are informed by the pilots and must be archived before new core runs. They are not descriptions of already observed results.

### H1 — Objective-specific channel crossover

The objective × intervention interaction is non-zero and stable. The anticipated ordering is that NM is most sensitive to neighborhood feature/coupling disruption, masked feature denoising to semantic disruption, and valid pair reconstruction to structural disruption; this ordering is a falsifiable hypothesis, not something guaranteed by the objective names. A universal fingerprint shared by all objectives falsifies H1.

### H2 — Objective-conditioned distances

`D_o(s→t)` improves held-out prediction of `Y_pre`, and `D_{o,q}(s→t)` improves held-out prediction of `Y_down`, over every objective-agnostic distance after controlling for source size, target difficulty, event family, language, missing features, and known identity overlap.

### H3 — Joint allocation is non-separable

Source–objective interaction explains held-out downstream variation beyond additive source and objective effects and produces replicated source-ranking crossovers. Under fixed compute, the channel-derived full `π_{s,o}` improves over rank-1/factorized and rank-2 policies with identical information, regularization, starts, and tuning budget, while shuffled-interaction controls do not.

### H4 — Robust generalist gain

The proposed joint mixture improves mean transfer and the worst held-out event family, with no predeclared family suffering a material regression relative to the strongest generic baseline. A bottom-quartile statistic is secondary and is used only if the study reaches at least eight genuinely independent families.

### H5 — Proxy-to-scale stability

Channel fingerprints, cell rankings, and selected weights estimated with the small model and short checkpoints remain predictive for the 5–10M-parameter model and longer budgets. The method need not preserve exact weights, but it must preserve decisions better than random or size-only allocation.

### H6 — Channel matching explains downstream benefit

An objective helps most when its fingerprint overlaps the downstream task-demand fingerprint and the source covers the target in those overlapping channels. A robust mixture succeeds by covering complementary task channels, not simply by lowering aggregate pretraining loss.

---

## 4. Method

### 4.1 Graphs and splits

The current catalog supplies eight source graphs: `ukraine`, `covid`, `midterm`, `covid-political`, `election2020-political`, `ukraine-suspended`, `twibot20`, and `hongkong`. Together the current source artifacts contain roughly **34.48M graph-node instances and 191.52M edges**. This is not a count of unique people; cross-event identity overlap remains to be audited. Two sources dominate node count, while six have fewer than 350k nodes.

Use [the graph catalog](../../config/graph_catalog.json) as the source of truth for names, paths, task support, and provenance. Do not describe all eight artifacts as identical retweet graphs: some include mention/retweet weights, missing-feature policies and task availability differ, and query counts vary.

The study needs four levels of exclusion:

1. **graph-key holdout** for continuity with current results;
2. **event-family holdout** so related full/political/suspended variants cannot straddle train and test;
3. **user-disjoint holdout** wherever stable identifiers permit an auditable overlap check;
4. **temporal holdout** for the large event streams.

Temporal slices of `covid` and `ukraine` can increase the number of development domains for fitting the geometry, but they are correlated subdomains, not independent external graphs. At least one later or separately collected social graph must remain untouched until the method and hyperparameters are frozen. Add it to the graph catalog only when its artifact and provenance exist.

### 4.2 Common encoder and compute accounting

The core comparison must give every objective access to a topology-capable encoder. The repository contains count-aware components, but the existing E2 configurations use one hop plus explicit directed-degree inputs; they are not already the required clean backbone. Gate 0 must therefore instantiate and validate a two-hop, count-aware path **without** target-degree passthrough before it becomes the primary encoder. Use a standard second backbone such as GIN for the compact generalization test. The one-hop mean-aggregation NM setup remains a legacy ablation because its sampled star and invariant aggregation prevent a strong topology claim.

Core scale:

- a roughly **2–3M-parameter** small model for the atlas and method search, recounted after the final encoder and heads are fixed (the legacy S/U/M model is ~1.64M; direct pre-head `sage_multi` instantiations are roughly 1.97M for one message-passing layer and 2.56M for the S2 variant);
- one **5–10M** scale point after the method is frozen;
- no 50M claim unless an actual scaling study later justifies it.

Use encoded node/edge tokens as the primary budget currency and verify the comparison with estimated encoder FLOPs. Convert each `π_{s,o}` into an objective-specific episode rate; different objectives need not receive equal update counts. For every treatment, also report optimizer updates, sampled nodes/edges, objective episodes, realized cell exposure, wall-clock time, and peak memory.

Heads should have comparable projection width. Report encoder parameters separately from objective-head parameters. Use the same optimizer family, scheduler, batch/episode node cap, and validation policy. Normalize per-head losses and gradient norms by a rule frozen in the pilot, not by inspecting final transfer.

### 4.3 Three primary pretraining objectives

Use three objectives with intentionally different information demands, while giving the core versions the same full `(A,X)` input so the channel audit—not the objective’s name—determines what they actually use.

1. **Neighbor matching (NM; candidate coupling/neighborhood-semantics objective).** Retain the current episodic task, but use within-source episodes and the common two-hop encoder. Evaluate both the current one-hop variant and the topology-capable variant.
2. **Masked node-feature denoising (MFD; candidate feature objective).** Fit whitening/standardization on the training fold, mask a fixed fraction of the target node’s transformed feature coordinates or blocks, and reconstruct only the hidden coordinates. Never pass a masked value through an identity path. Compare with feature-mean, PCA, and feature-only MLP reconstruction; add complete-node masking as a separate test of neighborhood coupling.
3. **Pair-conditioned edge reconstruction (Pair-LP; candidate structure objective).** Encode both queried endpoints or their joint enclosing subgraph after removing the queried edge. Score from `[h_u, h_v, |h_u-h_v|, h_u ⊙ h_v]`. The model must receive the identity/context of both endpoints. A constant-feature/structure-only view is an evaluator sanity control, not a substitute for the core full-input objective.

On the **small synthetic factorial only**, train every objective in four matched contexts: full input, `X`-only/edgeless, `A`-only/constant-feature, and coupling-broken with both marginals preserved. These training arms are the same cheap runs used to estimate the causal fingerprints. On the four natural pilot graphs, apply the corresponding inputs first as evaluation-time stress tests; launch additional natural training arms only if the synthetic-to-natural interpretation is ambiguous. The unrestricted full-input objectives remain the transfer comparison.

Future-link reconstruction can be added only after the pair-conditioned static evaluator passes its correctness gate. Contrastive learning can remain an appendix objective if its augmentation semantics are clean; three well-understood objectives are preferable to five poorly identified ones.

### 4.4 Gate 0: repair topology evaluation before training more objectives

The current static and temporal pair evaluators are not valid evidence for pairwise topology. They construct labels from a center–candidate relation while the model input is only the candidate-rooted subgraph; center identity is retained as metadata/debug state rather than as a scored endpoint. Consequently, below-chance orientation and the apparent MIX link-prediction gain cannot support topology claims.

The repaired evaluator must pass all of the following before any topology result enters the paper:

1. changing either queried endpoint while holding the other fixed changes the encoded input and score;
2. the positive edge is removed from the message-passing graph before encoding;
3. train/validation/test edges and temporal windows are disjoint under a documented policy;
4. negative sampling includes random, degree-matched, and hard two-hop negatives;
5. score orientation and threshold are selected on validation only and then locked;
6. common-neighbor, Adamic–Adar, preferential-attachment, raw-feature pair MLP, random encoder, and scratch-trained encoder baselines are reported;
7. a synthetic graph with known edge mechanism produces the expected baseline ordering;
8. an endpoint-permutation test destroys performance.

After repair, rescore all existing checkpoints before launching the atlas. This rescoring requires zero new pretraining, although evaluator sanity models and scratch/pair-MLP floors still require small fits.

### 4.5 Causal channel interventions

Use two complementary test beds.

#### Synthetic factorial test bed

Generate a crossed factorial of degree-corrected block graphs in which the node-feature marginal, unlabeled topology, and conditional feature–position binding are independently controllable. Verify that each operator holds the other declared marginals/statistics within a preregistered tolerance. For every objective, train matched-compute base and channel-removed arms, preserve query identities and clean evaluation targets, and evaluate all arms on the same uncorrupted held-out graph. This is the source of `Δ_{o,c,g,r}` and the strong causal language.

#### Natural-graph stress tests

Apply fixed, label-free interventions primarily at evaluation and, for a small mechanistic subset, during pretraining:

- **feature intervention:** mask or replace node features with equal-sample draws while keeping `A` fixed;
- **structure intervention:** degree- and community-aware rewiring while keeping `X` fixed;
- **coupling intervention:** permute `X` across nodes within degree/community strata, approximately preserving the marginals of `X` and `A` while breaking their binding;
- **temporal intervention:** shuffle or truncate interaction time where timestamps exist.

Natural interventions are approximate because feature replacement and rewiring generally alter coupling as well as a marginal. Report invariance diagnostics for every intervention, but use the synthetic conditional factorial effects—not natural corruption drops—to form `a_o`.

Fit a hierarchical factorial model with objective, removed channel, synthetic graph family, seed, and their planned interactions. The primary contrast is objective × channel. Retain graph-specific fingerprints and estimate a global `a_o` only if heterogeneity is below a preregistered stability bound; otherwise ChannelMix must use graph-conditional uncertainty or stop. Fit natural stress-test effects separately and never feed held-out transfer outcomes back into the fingerprint.

### 4.6 Channel distances and task demands

Predeclare one directional target-to-source coverage descriptor per channel and keep symmetric MMD/energy distances as sensitivity analyses. Let `n` be the largest common valid sample size up to 50,000, fixed after the data audit but before transfer outcomes are inspected.

- `d_X(t || s)`: mean target-to-source k-nearest-neighbor distance for `n` node features in a 64-dimensional PCA-whitened space fitted only on inner-training graphs, divided by the target’s split-half self-distance; zero/missing bios form a separately reported mass rather than feature vectors;
- `d_A(t || s)`: the same one-sided coverage statistic on a standardized local structural vector containing log in/out/total degree, local clustering, ego edge count/density, and a fixed compact heat-kernel signature;
- `d_XA(t || s)`: the same statistic on a single coupling vector consisting of edge-conditioned feature-similarity residuals relative to degree/language-matched nonedges, together with neighborhood-feature residuals after regressing out the structural vector on inner-training graphs.

Use identical sample counts and random seeds across graph pairs so a 23M-node source does not win coverage mechanically. PCA, standardization, matching strata, bandwidth/neighborhood choices, and any supervised calibration are recomputed inside each fold. The primary `D_o` is entirely outcome-free; an equally flexible transfer-outcome regression is a separate baseline, not part of the proposed distance. Include graph size, language, event family, construction pipeline, and missing-feature fraction as explicit controls.

Estimate task demand `b_q` on labeled development graphs from clean performance loss under crossed feature/adjacency inputs and from four mandatory floors:

1. raw node-feature model;
2. bag-of-neighbor-features/DeepSets model;
3. structure-only model;
4. random and scratch encoders.

Convert only stable positive channel effects to the nonnegative `b_q` simplex; retain negative shortcut effects separately and declare `b_q` undefined if its positive mass is indistinguishable from random variation. Recompute all task-demand estimates inside the outer fold and test their stability across development domains. This prevents a political-label task that is already solved by bios from being marketed as evidence of general graph reasoning.

### 4.7 The proposed method: ChannelMix

**ChannelMix** is a working name. It is a static joint mixture unless the allocation is explicitly allowed to change with training time; do not call it a curriculum merely because batches are interleaved.

ChannelMix is a **low-hyperparameter allocation heuristic**, not a theorem about how specialist utilities add inside one jointly trained encoder. It must first predict the ordering of held-out mixture trials in E3; otherwise stop before the full method sweep.

For development target–task cell `j=(t,q)`, define the outcome-free compatibility kernel

\[
K_{s,o,t,q}=\kappa_{o,q}\exp\!\left(-D_{o,q}(s\rightarrow t)/\tau\right).
\]

For objective-near transfer without downstream task `q`, use `exp(-D_o/τ)` and analyze it separately. If `a_o`, `b_q`, or their overlap is undefined, the corresponding cell receives no channel-based preference and falls back to its allocation floor.

#### Cross-fitting and data firewall

For every outer held-out family `f`:

1. remove every donor graph sharing event family, known users, or prohibited time windows with `f`;
2. hide every descriptor and outcome from `f` from the primary target-domain-agnostic policy;
3. on the remaining families, rotate an inner pseudo-target family `g`, again removing same-family/user/time donors and all self cells;
4. recompute whitening, structural standardization, matching strata, `b_q`, distance calibration, and all allocation hyperparameters inside that inner split; estimate `a_o` only from the independent synthetic training-time factorial data permitted to that fold;
5. give any outcome-calibrated baseline exactly the same inner transfer outcomes and tuning budget;
6. freeze one allocation over the outer fold’s eligible donor cells, train from scratch, and evaluate once on `f`.

RQ2’s geometry analysis is a separate unlabeled-target prediction test and may compute `D_o(s→f)`/`D_{o,q}(s→f)` from `f`’s descriptors. Those descriptors are never fed to the **primary** ChannelMix policy in the same outer fold. This separates an explanatory transfer law from an inductive generalist allocation.

For the final fully inductive external test, use every development family once as an excluded pseudo-target to create cross-fitted compatibility estimates, aggregate them by family, refit the few scalar hyperparameters, and emit one allocation over all eight development sources. Freeze it before opening the external graph. The secondary transductive version may then use external descriptors, but not labels, and is reported separately.

#### Allocation proxy and capacity-matched tests

Aggregate tasks within family before families are combined:

\[
z_f(\pi)=\frac{1}{|Q_f|}\sum_{q\in Q_f}
\log\left(\epsilon+\sum_{s,o}\pi_{s,o}K_{s,o,f,q}\right).
\]

Choose `π` from the cross-fitted kernels by maximizing mean family coverage plus a worst-family term and a uniform-prior regularizer:

\[
J(\pi)=\operatorname{mean}_f z_f(\pi)
+\gamma\min_f z_f(\pi)
-\lambda\operatorname{KL}(\pi\Vert u).
\]

Only τ, γ, λ, and the common allocation floor are tuned; no free cell utility is fitted from its own transfer outcome. The log/additive response is a stated heuristic assumption and is validated prospectively against actual mixtures.

Compare four policies with identical kernels, starts, regularization, and tuning budget:

1. **rank 1/factorized:** `π_{s,o}=p_s q_o`;
2. **rank 2:** nonnegative two-component source–objective allocation;
3. **full channel policy:** unrestricted `π` chosen from the outcome-free kernel;
4. **interaction controls:** full policies using universal distance, shuffled fingerprints, and source–objective interaction-shuffled kernels.

Replicated source-ranking crossovers in the common downstream tensor plus a full-over-rank-1 mixture gain are both required for non-separability. A specialist interaction alone does not prove joint-training interaction, and extra allocation flexibility alone does not prove the channel account.

The executable generic baseline is a GRAPE-style group-DRO reweighter over graph–objective cells using development validation losses; it receives the same inner outcomes and tuning budget as any calibrated ChannelMix variant. The objective-only baseline is ControlG-style scheduling with uniform sources, and the source-only baseline uses the same channel descriptors with uniform objectives.

If checkpoint analyses show stable early/late changes in channel sensitivity, a two-stage **schedule** can be tested after the static method. Call it a curriculum only if it beats reversed and time-permuted schedules with the same integrated allocation.

### 4.8 Sampling and episode construction

The source/objective allocation does not fix how an episode is constructed. Run the following 2×2 factorial on representative large–small and related–unrelated source pairs:

| | Globally pooled candidates | Within-source candidates |
|---|---|---|
| **size-proportional source exposure** | naive merged baseline | isolates candidate shortcut from exposure |
| **source-balanced exposure** | isolates balancing from confinement | current strongest sampling design |

Log realized source counts and source predictability from episode metadata. A globally pooled NM episode may admit domain/source shortcuts that disappear at single-domain evaluation; within-source construction and balancing are distinct interventions and must not be conflated.

For single-objective source-composition claims, where an update has the same cost in every arm, compare:

1. best single source for `U` updates;
2. best single source for `kU` updates;
3. `k`-source mixture for `U` total updates;
4. `k`-source mixture for `kU`, approximately matching per-source exposure;
5. one larger-capacity `k`-source mixture at `U`.

Only these controls can separate complementarity from exposure dilution and capacity/optimization interference.

### 4.9 Evaluation tasks and floors

Use a declared evaluation panel rather than whichever metrics happen to look favorable.

#### Pretext-near tasks

- NM 30-way/3-shot accuracy and ROC-AUC;
- repaired pair-conditioned static link prediction;
- temporal link prediction only where temporal provenance is valid.

#### Downstream tasks

- node classification on the four catalog-supported labeled graphs;
- profile regression on the supported graphs, treated as secondary unless it beats raw-feature and scratch floors;
- at least one topology-required synthetic task whose label is independent of raw bios;
- few-shot linear probing as primary, with a compact fine-tuning check on two held-out targets.

For every target–task cell report raw performance for:

- majority/chance or heuristic floor;
- raw features;
- bag of neighborhood features;
- structure only;
- random encoder;
- scratch model with the same architecture and label budget;
- source/objective specialist;
- all proposed mixtures.

### 4.10 Primary metrics and statistics

Raw metrics remain primary. For cross-task aggregation, use specialist-normalized retained headroom only when the denominator is stable:

\[
R_{t,q}(m)=
\frac{m_{t,q}-m^{random}_{t,q}}
{m^{specialist}_{t,q}-m^{random}_{t,q}}.
\]

If the specialist fails to beat random by a predeclared minimum margin, do not normalize that cell; flag it as an invalid denominator.

Primary summary outcomes:

1. mean `R` after weighting tasks equally within each held-out event family;
2. worst-held-out-family `R` and raw performance;
3. bottom-quartile/CVaR-.25 only if at least eight genuinely independent families become available;
4. transfer-prediction rank correlation and regret within target, never pooled across repeated target difficulty;
5. compute–performance curves at 5k, 10k, 20k, 40k, and 80k updates.

Use three training seeds for headline models. Bootstrap evaluation queries within run and report seed variation separately. Treat graph/event family—not individual query episodes—as the unit supporting cross-domain generalization. Use nested leave-family-out evaluation for all learned distances and allocation hyperparameters. Correct the small planned family of confirmatory interaction tests; label the rest exploratory.

---

## 5. Experiment program and realistic run budget

The program is gated so a failed premise does not generate hundreds of uninformative runs. All heavy training belongs on Tucker GPUs 0–3; the user should launch long jobs in tmux after the code is committed and pulled. Prefer a multi-artifact loader over materializing additional 100GB merged graphs.

### E0 — correctness and evidence repair (no new social-graph pretraining)

- implement and unit-test the pair-conditioned evaluator;
- rescore existing NM/FP/CL/MIX/E2/E4 checkpoints;
- add all raw/bag/structure/random/scratch floors;
- audit event lineage, user overlap, label availability, and temporal splits;
- freeze the graph-family map, metrics, and intervention definitions.

**Go/no-go:** do not run a structural atlas until Gate 0 passes.

Frozen-checkpoint rescoring needs no new pretraining, but pair-MLP/scratch floors and synthetic evaluator validation require small supervised fits; log them separately rather than calling the whole gate “zero training.”

### S0 — synthetic causal fingerprint

Before natural graph training, run three small synthetic graph families × three objectives × four contexts (full, `X`-only, `A`-only, coupling-broken) × three seeds = **108 short synthetic jobs**. These estimate the paired training-time factorial effects and validate the channel anchors. They are not 40k social-graph equivalents; measure and report their actual node-token/FLOP cost separately.

**Go/no-go:** stop the channel story if the interventions fail their invariance checks or objective × channel crossover is not reproducible across synthetic families.

### E1 — four-graph channel pilot (first 12 atlas cells)

Select four deliberately different graphs: one massive broad donor, one related small graph, `twibot20`, and `hongkong`. Train 4 graphs × 3 objectives × 1 seed using the final common architecture, and evaluate the natural stress tests. These 12 runs become seed 0 of the atlas if the gate passes.

**Go/no-go:** continue only if at least two objectives have reproducibly different top-ranked channel sensitivities on the synthetic test bed and the objective × intervention pattern is directionally consistent on at least three of four natural graphs. Replicate the 12 cells for seeds 1–2 before calling the fingerprint stable.

### E2 — objective-conditioned transfer atlas

Complete 8 sources × 3 objectives × 3 seeds = **72 specialist runs** total. The current eight one-hop NM specialists remain legacy evidence rather than being mixed into the primary two-hop atlas. Evaluate every checkpoint on every valid held-out graph and task without retraining.

Fit universal and objective-conditioned transfer models only through nested event-family folds.

**Go/no-go:** proceed to a method claim only if `D_o` improves held-out `Y_pre` prediction and `D_{o,q}` improves held-out `Y_down` prediction over size-only, event-family, and universal-distance baselines, with neither result driven solely by `covid`/`ukraine`.

### E3 — exposure and episode controls

Use atlas results to select two diagnostic source sets rather than repeat every mixture:

- a large–small related pair;
- a broad–isolated or four-source set.

Run the 2×2 sampling factorial, the `U` versus `kU` exposure controls, and one capacity control. Reserve at least six runs for actual mixtures chosen to span ChannelMix’s predicted ordering; this prospectively tests the log/additive proxy before E4. Replicate the decisive sampling/exposure and proxy-ordering cells for three seeds. Budget **18–30 new runs**, depending on which existing seed-0 conditions are protocol-compatible.

**Go/no-go:** stop the method if the outcome-free kernel cannot rank these held-out mixture trials better than universal-distance or shuffled-fingerprint controls.

### E4 — ChannelMix versus alternatives

Let `F` be the number of independent event-family folds established by E0; the catalog does not currently encode this map, so do not hard-code it in advance.

Seed-0 screen: 8 methods × `F` folds = **8F runs**:

1. sequential graph/objective blocks;
2. size-proportional interleaving;
3. uniform joint mixture;
4. source-only optimized mixture with uniform objectives;
5. objective-only scheduler with uniform sources;
6. optimized factorized mixture;
7. generic outcome-driven joint reweighting;
8. ChannelMix.

Then replicate ChannelMix, factorized ChannelMix, and the strongest non-ChannelMix baseline for seeds 1–2: 3 methods × `F` folds × 2 seeds = **6F runs**. Total **14F** if the screen passes (56 runs for four folds; 84 for six).

**Go/no-go:** the method contribution requires a replicated gain over factorized ChannelMix in mean and worst-family transfer, not merely a win over sequential training.

### E5 — scale and external validity

After freezing all choices:

- compare the top three methods on a 5–10M model;
- repeat a compact subset on a second backbone;
- evaluate on one untouched external/later graph;
- produce checkpoint scaling curves;
- fine-tune on two downstream targets as a secondary check.

Budget **12–18 runs**. Do not expand scale if E4 fails.

### Total

The full successful route is **`102–120 + 14F` new 40k-equivalent social-graph runs**, plus 108 much smaller synthetic jobs, evaluator/baseline fits, and evaluation-only sweeps. This is 158–176 social runs if E0 yields four independent folds and 186–204 if it yields six. Protocol-compatible existing seed-0 controls may reduce the number of new jobs, but should not be mixed into the core study merely to save compute. The program is realistic only because it is staged, most runs use the small model, all 72 atlas models are embarrassingly parallel, and the expensive scale study happens last. Measure representative social and synthetic runs before promising a calendar schedule.

---

## 6. Results overview

### 6.1 Results already in hand

| Finding | Status | Current result | Safe use in v2 |
|---|---|---|---|
| Single-source NM transfer is heterogeneous and directional. | **[EXISTING — 1 seed]** | Complete 8×8 matrix. Specialist diagonals are .906–.981 AUC; broad-donor and narrow-specialist rows differ substantially. | Motivation for a transfer geometry. Identity/event overlap still needs auditing, so do not call every off-diagonal “zero-overlap.” |
| A fixed-budget target-excluded NM mixture is close to its best constituent donor. | **[EXISTING — 1 seed]** | Across the 21 true multi-source, target-excluded ladder cells: mean absolute difference .0073 AUC, signed mean −.0044, max absolute .0201; no gain >+.010 and six losses <−.010. | Shows little **net** gain under the current order and 40k total budget. It does not establish zero synergy because source exposure is diluted. |
| Exact target entry creates large jumps for isolated graphs. | **[EXISTING — 1 seed]** | `covid-political` +.081, `election2020-political` +.096, `ukraine-suspended` +.165, `hongkong` +.140 AUC when each enters the ladder; `twibot20` is +.013. | Separates exact-target inclusion/exposure from foreign-source generalization. |
| Inclusive all-eight NM pays an in-domain tax. | **[EXISTING — 1 seed]** | Specialists exceed all-eight by .006–.039 AUC on all eight graphs. | Evidence of fixed-budget dilution/interference, especially on small graphs; replicate. |
| Source-aware sampling rescues a small source. | **[EXISTING — 1 seed]** | In the `covid`/`midterm` setting, proportional sampling leaves `midterm` near .31–.33 NM accuracy; balanced within-source sampling reaches .405 at matched compute and .427 at higher compute; the `midterm` specialist is .417. | Strong motivation for episode accounting. Balancing and within-source confinement are currently confounded, hence the 2×2. |
| One-hop NM is driven by real neighborhood feature content. | **[EXISTING — 1 checkpoint/seed]** | `covid` accuracy: intact .664, within-neighborhood feature permutation .626, zero .073, sampled noise .061. `midterm` and `twibot20` show the same high intact/permuted versus low zero/noise pattern (not the exact same within-pair ordering). | Establishes that the current solution uses the real feature multiset delivered by the neighborhood. It does not show that a topology-capable model chose to ignore rich topology. |
| A simple target router is not yet needed. | **[EXPLORATORY]** | The largest-available-source policy chooses the oracle foreign donor for 8/8 targets on NM accuracy and 7/8 on AUC, with ~.0004 AUC regret; the feature/proxy router chooses 5/8 with ~.0146 regret. | Do not claim source-selection novelty. Objective-conditioned geometry needs harder and external targets. |
| Different valid node tasks show objective specialization. | **[EXPLORATORY — 1 seed]** | In the rotation study, NM/MIX classification AUC is .810/.795 over two held-out labeled graphs; FP/MIX regression Spearman is .166/.097 over four graphs × three common targets. | Weak motivation that objectives expose different useful information. The topology column from this experiment is quarantined. |
| Broad frozen-representation utility is not established. | **[EXPLORATORY/negative]** | In a separate 23-cell probe matrix using different checkpoints, raw features average about .109 Spearman, random encoder about .022, and current trained rows do not beat the raw baseline. Political classifications are also strongly predictable from raw bios. | Forces common floors and a channel-matched downstream panel. Do not compare these means directly with the rotation panel. |
| Sequential pretraining is “bad.” | **[NOT SUPPORTED/partly quarantined]** | The current aggregate mixes three invalid LP evaluation columns, bootstraps benchmark cells rather than training seeds, and changes when LP columns are removed. There is no clean seed-replicated effect estimate. | Keep sequential training as a future matched-compute baseline; do not report the legacy point magnitudes. |
| “90% of performance in 10% of steps.” | **[NOT SUPPORTED]** | No validated current analysis supports this sentence. | Replace with prospectively measured checkpoint curves. |

Current evidence sources: [single-source NM findings](../../scripts/experiments/analysis/nm_single_source_matrix/FINDINGS.md), [NM ladder results](../../scripts/experiments/analysis/nm_ladder/RESULTS.md), [covid/midterm exposure results](../../scripts/experiments/analysis/nm_covid_midterm/RESULTS.md), [feature-ablation findings](../../scripts/experiments/analysis/feature_ablation/FINDINGS.md), [mixed-objective lattice, valid metrics](../../scripts/experiments/analysis/multitask_ssl/FINDINGS.md), [probe-matrix findings](../../scripts/experiments/analysis/pretrain_probe_matrix/FINDINGS.md), and the superseding [directed/log-input objective rerun](../../scripts/experiments/analysis/topology_feature_ssl/RESULTS_directed3log.md). The 21-cell best-donor and router summaries are checked derivations currently narrated in the coverage v1; add a reconstruction script and committed table before citing them in a manuscript.

### 6.2 Quarantined results

| Result | Why quarantined | Disposition |
|---|---|---|
| **Every current static and temporal LP result**—including pretrain-probe NM→static-LP .612 and LP-containing sequential aggregates | The legacy evaluators do not supply a valid scored endpoint pair, and some aggregate analyses mix these columns with valid node tasks. | Exclude from all filled headline quantities; rerun after Gate 0. |
| MIX static-link AUC .759 and the apparent “emergent” three-objective result | The evaluator does not condition the prediction on both queried endpoints. Pair identity is therefore absent from the model input in the intended sense. | Mention only here. Rescore after Gate 0. Even if it survives, treat it as supporting/appendix evidence rather than the paper thesis. |
| The three-way/pairwise objective lattice | Same invalid pairwise topology read, one seed, and no common floors. | No headline or hypothesis is built around it. |
| E1/E2/E4 topology and regression headlines | Current pair evaluator cannot establish pairwise topology; some arms contain explicit/composite changes. In addition, `RESULTS_directed3log.md` shows that the original E1 regression and E2 LP headlines were input-scaling artifacts. | Use the superseding rerun for diagnosis only; rebuild clean arms under the common encoder and corrected endpoint task. |
| Below-chance AUC interpreted as “no signal,” or flipped after test inspection | AUC below .5 can contain reversed signal—e.g. CLFP .227 becomes .773 after a post-hoc flip—but post-hoc orientation is not a valid score. | A flip is diagnostic only. Lock orientation/calibration on validation and report the untouched test result. |

The quarantine is deliberately blanket rather than selective: **no current static/temporal LP number supports the v2 paper.** The paper does not depend on the uncertain three-way result.

### 6.3 Required result slots

These comments are the skeleton of the eventual Results section. Fill numbers, uncertainty, and failures; do not replace them with qualitative adjectives.

#### RQ1 — Channel fingerprints

**[OUTSTANDING]** Objective × intervention interaction on the synthetic factorial benchmark: `[effect, 95% CI, corrected p-value]`.

**[OUTSTANDING]** Replicated synthetic causal fingerprints `a_o` plus separately labeled natural-graph reliance stress profiles: `[table/figure; seed, synthetic-family, and natural-graph variation]`.

**Decision comment:** if all objectives share the same fingerprint, stop the ChannelMix story. If natural and synthetic results disagree, frame the synthetic result as identifiability and investigate the natural confound before proceeding.

#### RQ2 — Objective-conditioned geometry

**[OUTSTANDING]** Nested leave-family-out prediction of `Y_pre` using `D_o`, and of `Y_down` using `D_{o,q}`, against size-only, universal-distance, graphon/OT, shuffled-fingerprint, and equally flexible outcome-only models: `[within-target Spearman, MAE, top-source regret, uncertainty]`.

**[OUTSTANDING]** Source × target × objective transfer atlas: `[8×8 panels per objective, three seeds]`.

**[OUTSTANDING]** Task-demand stability and channel-overlap rule: `[cross-domain stability of b_q; product-rule D_{o,q} versus objective-only/task-similarity alternatives]`.

**Decision comment:** `D_o` must improve out-of-family prediction, not merely fit the observed 8×8 matrix. Report whether `covid`/`ukraine` remain trivial universal donors.

#### RQ3 — Non-separability

**[OUTSTANDING]** Source × objective interaction on the common downstream tensor and replicated source-ranking crossovers: `[hierarchical interaction estimate / held-out likelihood gain / crossover count]`.

**[OUTSTANDING]** Rank-1, rank-2, full joint, universal-distance-full, and interaction-shuffled allocations at fixed node-token/FLOP budget: `[mean ΔR, worst-family ΔR, per-family deltas, CIs]`.

**Decision comment:** if the factorized mixture ties the full mixture, the graph–objective pair is not an empirical method contribution even if objective-conditioned distances remain scientifically useful.

#### RQ4 — Generalist performance

**[OUTSTANDING]** Main method table for sequential, proportional, uniform, source-only, objective-only, factorized, generic joint, and ChannelMix: `[raw metrics + normalized mean/worst family]`.

**[OUTSTANDING]** Exposure- and sampling-controlled decomposition: `[balancing effect, within-source effect, interaction, U vs kU]`.

**Decision comment:** require both mean and worst-family improvement. A mean win produced by sacrificing `hongkong` or a small labeled graph is not a robust generalist result.

#### RQ5 — Scale and external target

**[OUTSTANDING]** Small-to-5–10M rank/weight stability and checkpoint curves: `[correlation, regret, compute efficiency]`.

**[OUTSTANDING]** Second-backbone and untouched external/later graph result: `[raw and normalized metrics]`.

**Decision comment:** if the method works only on the development graphs or one custom backbone, narrow the claim to a social-graph benchmark study.

---

## 7. Actual contributions

### 7.1 Evidence-backed assets that exist now

As of 2026-07-22, the repository supports four concrete assets—not yet four finished paper claims:

1. a complete one-seed 8×8 single-source NM transfer matrix and eight-rung inclusion ladder;
2. a fixed-budget analysis separating target membership, strongest-donor consolidation, and mixture dilution (the audited 21-cell reconstruction still needs its own checked-in script/table);
3. a diagnostic intervention showing that current one-hop NM requires real neighborhood feature content;
4. a sampling pilot showing that naive source exposure can severely undertrain a small graph.

These justify the v2 experiment. Seeds, holdout integrity, common floors, and a valid pair evaluator are still required.

### 7.2 Contribution statements the finished paper is designed to earn

These are the clear contribution claims for the full paper. Delete any item whose acceptance test fails.

1. **Causal principle — objective-specific transfer channels.** We introduce a training-time synthetic factorial protocol that identifies conditional feature, structure, and coupling effects, then test whether those fingerprints explain natural-graph reliance and transfer.  
   **Acceptance test:** H1 passes across seeds and synthetic families, intervention invariances hold, and the separately labeled natural stress tests are consistent.

2. **Empirical object — an objective- and task-conditioned transfer atlas and geometry.** We provide a source × objective × target × task transfer tensor, separate objective-near from downstream outcomes, and show that independently estimated channel discrepancies predict held-out event-family transfer better than a universal graph distance or graph size alone.  
   **Acceptance test:** H2 passes in nested leave-family-out evaluation and on the external graph.

3. **Method — ChannelMix.** We propose a cross-fitted, channel-aware allocation heuristic for the joint graph–objective compute distribution.  
   **Acceptance test:** it predicts held-out mixture ordering and improves mean and worst-family transfer over rank-1/factorized, rank-2, shuffled-channel, universal-distance-full, and generic joint baselines under matched node-token/FLOP compute; otherwise this is not a contribution.

4. **Benchmark resource — audited social-graph transfer.** If released as a reusable suite, we contribute the objective-conditioned atlas, event-family/user/time split map, pair-conditioned tasks, common raw/bag/structure/random/scratch floors, and provenance needed to distinguish semantic shortcuts from structural transfer.  
   **Acceptance test:** the split/evaluator/floor suite is documented and reusable beyond this paper. A corrected pair evaluator by itself is protocol hygiene, not a contribution.

5. **Practical finding — budget, exposure, and scale robustness.** We quantify when apparent multi-source/objective gains are coverage, balanced exposure, complementarity, or dilution, and test whether proxy allocations survive one larger model, longer budgets, and a later event.  
   **Acceptance test:** the `U/kU`, 2×2 sampling, checkpoint, and scale controls are completed.

### 7.3 Claims that are explicitly not contributions

- a generic statement that “pretext tasks matter”;
- generic source selection or generic objective scheduling;
- the largest source being a strong NM donor;
- the current uncertain three-way “emergence” result;
- “best” performance without a fixed comparison set and confidence interval;
- a 50M model or released weights that do not yet exist;
- calling an inclusive target-trained score zero-shot transfer;
- calling a static mixture a curriculum.

---

## 8. Positioning against adjacent work

The closest literatures already occupy much of the naive claim space:

| Prior direction | What it establishes | What v2 must add |
|---|---|---|
| [PRODIGY](https://arxiv.org/abs/2305.12600) | In-context graph pretraining and prompt-graph objectives. | Cross-social-graph data/objective allocation and channel identification. |
| [GSTBench](https://arxiv.org/abs/2509.06975) | Graph SSL objectives differ in cross-dataset transfer. | Causal channel fingerprints, source-dependent objective geometry, and a joint mixture. |
| [When Do Graph Foundation Models Transfer?](https://arxiv.org/abs/2605.29828) | Data-centric theory for cross-domain output shift and structural discrepancy. | Objective-conditioned rather than universal discrepancy, tested under feature/structure/coupling interventions. |
| [ParetoGNN](https://arxiv.org/abs/2210.02016), [WAS](https://arxiv.org/abs/2403.01400), and [ControlG](https://arxiv.org/abs/2602.05036) | Multi-objective graph SSL, task weighting/selection, and temporal objective allocation. | The source graph and objective are jointly valued; factorized versus non-separable allocation is the central test. |
| [AutoSSL](https://openreview.net/forum?id=rFbR4Fv-D6-) | Automated graph SSL objective selection. | Source-dependent objective value and causal channel measurements, not objective search alone. |
| [GPPT](https://doi.org/10.1145/3534678.3539249) and [task-similarity/Bridge-Tune](https://ojs.aaai.org/index.php/AAAI/article/view/29156) | Pretext/downstream alignment and task similarity. | Independently identified channel overlap plus cross-domain source geometry; compare against task-similarity baselines. |
| [APT](https://proceedings.neurips.cc/paper_files/paper/2023/hash/b29adb4bf2364acec8fb402ef731bb3b-Abstract-Conference.html), [W2PGNN](https://arxiv.org/abs/2303.16458), [SelMAG](https://arxiv.org/abs/2406.10425), and [GRADATE](https://papers.neurips.cc/paper_files/paper/2025/hash/7b69bc53449ba46bb981951078929a5e-Abstract-Conference.html) | Graph pretraining data/source selection and transferability modeling. | Selection at the graph–objective cell level, explained by outcome-free channel measurements; target-aware selectors belong in the secondary transductive comparison. |
| [LAMP](https://openreview.net/forum?id=rFRerAPdwI) and [GPH²](https://arxiv.org/abs/2602.13075) | Multi-source graph pretraining and task-oriented source fusion. | Fixed-compute source–objective non-separability with rank-controlled allocation tests. |
| [Topology Only Pre-Training](https://arxiv.org/abs/2311.03976) | Feature-independent topological pretraining can transfer across domains. | A measured feature/structure/coupling comparison inside one social-graph transfer design. |
| [GRAPE](https://arxiv.org/abs/2505.20380) | Generic joint source-domain and target-task reweighting for robust pretraining. | Graph-specific causal channels, objective-conditioned cross-graph geometry, and comparison against a generic outcome-driven joint baseline. |

The novelty sentence should therefore be precise:

> Prior work studies which graph data to select or how to combine graph objectives; we study why the value of graph data changes with the objective, identify the responsible information channel through interventions, and optimize the resulting non-separable graph–objective mixture for held-out transfer.

---

## 9. Paper structure and decisive figures

### Proposed narrative

1. **Introduction:** heterogeneous graph pretraining has two entangled allocation axes—data and objective.
2. **Audit:** current NM transfer and feature ablations show that one source ranking is channel-specific.
3. **Channel identification:** synthetic and natural interventions recover objective fingerprints.
4. **Transfer geometry:** objective-conditioned distance explains the transfer tensor.
5. **ChannelMix:** full joint allocation versus factorized and generic alternatives.
6. **Evaluation:** event-family holdouts, downstream channel panel, exposure/sampling, scale, and external graph.
7. **Limitations:** operational channels, social-media scope, correlated event variants, and remaining evaluator boundaries.

### Figures worth building

1. **Graph–objective grid:** encoded-node/edge compute budget over cells, contrasting sequential, factorized, and full joint allocations.
2. **Channel fingerprints:** objective × intervention heatmap with graph/seed uncertainty.
3. **Objective-conditioned atlases:** one 8×8 transfer panel per objective plus the failure of a universal geometry.
4. **Non-separability:** full joint versus factorized weights and performance on the mean/worst-family frontier.
5. **External validity:** proxy-to-scale and development-to-external transfer.

### Tables worth building

1. graph construction, size, features, tasks, family, identity/time overlap;
2. repaired evaluator and trivial floors;
3. main method comparison with raw and normalized metrics;
4. ablations of channel fingerprints, distance terms, sampling, and exposure;
5. per-family failure table so the mean cannot hide regressions.

---

## 10. Failure modes and paper pivots

This plan is valuable only if it is falsifiable.

### If H1 fails

Do not run ChannelMix. The paper may become an evaluator/channel-audit paper if it uncovers that apparently different objectives all exploit the same social-feature shortcut. That result needs strong synthetic controls and repaired topology evaluation to be significant.

### If H1 passes but H2 fails

Publishable mechanism may remain, but drop the transfer-prediction claim. Investigate whether channel descriptors are too coarse, source size dominates, or eight graphs are insufficient.

### If H1/H2 pass but H3 fails

The scientific contribution becomes objective-conditioned transfer geometry; use a simple factorized mixture in practice. Do not present ChannelMix as superior.

### If ChannelMix improves mean but hurts the worst family

It is a specialization method, not a robust generalist. Either change the declared deployment goal or report the trade-off honestly.

### If the external graph reverses the result

Narrow the scope to the eight-domain benchmark and analyze the missing channel. Do not average the external failure away.

### If repaired Pair-LP eliminates the old topology result

That is expected under the current audit and does not damage the paper. The new story begins from valid objectives and does not rely on the three-way pilot.

---

## 11. Reproducibility and scope constraints

- Keep each experiment atomic under `scripts/experiments/setup/<name>/`; place findings, notebooks, committed result CSVs, and figures in the matching analysis tree.
- Read graph metadata from `config/graph_catalog.json`; do not create another hard-coded graph registry.
- Use the shared train/eval harness and expose source/objective weights, seed, budget, node cap, and sampling policy as config/CLI parameters.
- Store exact realized graph–objective episode counts with every run.
- Run dry-run manifests before launching sweeps.
- Use `/dataMeR1/phil/data` and `/dataMeR1/phil/gfm/prodigy` on Tucker; do not assume those paths exist locally.
- Use GPUs 0–3 only. The user normally launches long jobs; provide exact tmux commands after implementation rather than silently starting them.
- Archive the hypotheses, folds, exclusions, metrics, and stop rules before E1.
- Release weights only if licensing permits and the final encoder beats scratch/raw floors on the declared panel. Weights are an artifact, not the scientific contribution.

---

## Bottom line

The better paper is not “we tried several curricula and found the best one,” and it is not “three objectives unexpectedly create topology.” It is:

> **Graph pretraining data need not have objective-independent value. When objectives expose different feature, structure, and coupling channels, those channels reshape cross-graph transfer and make a joint graph–objective allocation preferable under the tested fixed-compute budget.**

The current NM matrix, membership jumps, sampling result, and feature intervention give this story a credible starting point. The repaired evaluator, causal channel crossover, objective-conditioned out-of-family prediction, and full-joint-versus-factorized test are what would make it novel, significant, citable, and real.
