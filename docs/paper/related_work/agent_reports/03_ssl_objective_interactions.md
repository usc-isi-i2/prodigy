# Lit sweep 3/5 — SSL objective interactions, multi-task SSL, evaluation rigor

*Deep literature agent report, 2026-07-25. Post-Jan-2026 arXiv swept explicitly. Verbatim.*

## Closest papers (ranked by threat/relevance)

### 1. ControlG — detailed subsection
**"Feedback Control for Multi-Objective Graph Self-Supervision"** — arXiv **2602.05036** — **Feb 4, 2026 (post-cutoff)** — Grover, Vasiloudis, Xie, Lu, Song, Faloutsos (Amazon Science).
- **Claims:** Combining graph SSL objectives fails via three failure modes — *Disagreement* (conflict-induced negative transfer), *Drift* (nonstationary objective utility), *Drought* (starved objectives). Fix: treat coordination as **temporal allocation** — per-objective difficulty + pairwise antagonism estimation, Pareto-aware log-hypervolume planning, and a PID controller that decides *when* each objective gets budget.
- **Setup:** 5 pretext tasks (link prediction, masked feature reconstruction, node–subgraph MI contrast, decorrelation, METIS partition prediction — essentially ParetoGNN's task set), 9 small/medium benchmarks (Cora→ogbn-arxiv). **Frozen encoder + linear probe / logistic LP decoder / K-means** — *not* episodic, not few-shot, not in-context.
- **Results:** ControlG avg rank 1.4 vs best single objective (p_recon, 6.8) on node classification; on heterophilic graphs only "competitive with the best single-pretext method." Crucially: **random scheduling often matches or beats sophisticated multi-task weighting** — i.e., per-update gradient mixing itself causes interference.
- **Scoop-risk (honest):** **Moderate-high on framing, low on substance.** They now own "combining graph SSL objectives interferes; scheduling matters." But their headline is the *opposite conclusion* to ours: a well-scheduled combination *wins*. They never test whether the task-aligned single objective beats the combination on LP specifically, never test episodic/in-context frozen evaluation, never test sequential-order forgetting, never test web-scale social graphs, and never use heuristic floors or degree-matched negatives. Our (b) reads as the **boundary condition ControlG doesn't reach**: even temporal separation (rotation) dilutes when one objective is strictly downstream-aligned. RELATION: **must-cite + partial scoop of framing; we are the counter-evidence to "scheduling rescues combination."**

### 2. GSTBench — arXiv **2509.06975** — Sept 2025 — Song, Hua, Xie, Liu, Long, Liu
Benchmarks transferability of 5 SSL objectives (GraphMAE, VGAE, DGI, GRACE, LP) pretrained on ogbn-papers100M; downstream 5-shot NC + LP with linear probe / ICL / fine-tuning. Findings: **no universal winner; LP pretext best for LP downstream; contrastive methods show negative transfer; raw LLM features are competitive baselines.** RELATION: **must-cite, partial scoop of finding (a)** at single-objective level — but no objective *combination*, no regression task, no heuristic floors, no dilution/forgetting story.

### 3. ParetoGNN — ICLR 2023 — arXiv **2210.02016** — Ju et al.
5 pretext tasks, multiple-gradient descent to "minimize potential conflicts"; claims multi-task SSL yields **stronger task generalization** across NC/clustering/LP/partition on 11 datasets. RELATION: **contradicts our finding (b) — we'd be the counter-evidence**; their own reliance on MGDA implicitly concedes conflict exists.

### 4. AutoSSL — ICLR 2022 — arXiv **2106.05470** — Jin et al.
Learns loss weights over SSL tasks via pseudo-homophily; shows different pretext tasks help different datasets ("searching pretext tasks is crucial"); downstream = clustering + NC only, no LP. RELATION: **must-cite**; motivates weighting but claims weighted combination beats singles.

### 5. WAS — ICLR 2024 — arXiv **2403.01400**
"Decoupling Weighing and Selecting" — instance-level task *selection* (not all tasks should be used) + weighting via siamese networks; 16 datasets. RELATION: **must-cite**; closest prior admission that naive inclusion of all tasks hurts, still concludes combination wins.

### 6. HeaRT — NeurIPS 2023 D&B — arXiv **2306.10453**
LP benchmarks use trivially easy random negatives; hard heuristic-related negatives collapse GNN advantages. RELATION: **methodological anchor** for our heuristic floors + hard negatives.

### 7. Implicit degree bias in LP — arXiv **2405.14985** — Aiyappa et al. 2024
Standard LP evaluation is degree-biased to the point a **pure degree baseline is near-optimal**; proposes degree-corrected benchmark. RELATION: **methodological anchor** — degree-matched negatives are *proposed*, not standard; directly legitimizes our evaluator repair.

### 8. EdgeBank — NeurIPS 2022 D&B — arXiv **2207.10128**
Trivial memorization baseline beats dynamic-LP SOTA under weak negatives → new eval protocol. RELATION: **precedent that evaluator-forensics + repair is a publishable contribution class.**

### 9. Pitfalls in LP with GNNs (target-link inclusion) — arXiv **2306.00899**
Documents leakage (test edge present in message passing) in *published* pipelines. RELATION: **must-cite precedent** for "silently broken published evaluators"; none of these cover PRODIGY-style center-blind episodic scoring.

### 10. GraphFM benchmark — arXiv **2406.08310** — 2024
8 GSSL methods: none generalizes across NC/LP/clustering simultaneously; "may not effectively serve as foundation models." RELATION: must-cite support for (a).

### 11. AUG-MAE ("Rethinking Graph MAE through Alignment and Uniformity") — AAAI 2024
GraphMAE avoids full collapse but suffers **partial dimensional collapse** (low-rank subspace). RELATION: must-cite for (c) — collapse under masked-feature objectives is known in principle.

### 12. Transductive Linear Probing — LoG 2022 — arXiv **2212.05606** — Tan et al.
Simple linear probes on SSL embeddings beat supervised graph meta-learners in few-shot NC. RELATION: adjacent evaluation-rigor precedent ("your fancy few-shot pipeline loses to a simple baseline").

Also relevant, lower threat: **Evaluating Progress in GFMs** (arXiv **2603.10033**, submitted Feb 28, 2026 — post-cutoff) — 8 GFMs, 33 datasets; *explicitly excludes PRODIGY*; finds no GFM dominates and single-domain pretraining sometimes beats multi-domain. **OpenRFM** (arXiv **2606.04320**, **June 2026**) — diagnoses relational (Kumo-style, not PRODIGY) ICL failures under sparse label coverage; adjacent evidence that ICL can silently fail. **GCL evaluation pitfalls** (arXiv 2402.15680); **Akhondzadeh et al., "Probing Graph Representations," AISTATS 2023**; **MolGraphEval** (arXiv 2206.08005).

## Answers to the six questions

**1. Multi-pretext graph SSL.** AutoSSL/ParetoGNN/WAS all claim a (weighted/Pareto/selected) combination beats singles, on NC/clustering/LP/partition over small-medium benchmarks; conflict is acknowledged only as an optimization nuisance to be solved. ControlG (2602.05036) is the first to elevate interference to the headline ("Disagreement/Drift/Drought") and shows random *temporal* scheduling beats gradient mixing — but still concludes scheduled combination wins. **No paper reports our result: combination strictly diluting the best single objective on its aligned task, or catastrophic forgetting between SSL *objectives* on graphs, or interleaving rescuing multi-graph but not multi-objective training.** Continual-SSL work (e.g., 2104.12081, 2205.09357) claims SSL forgets *less* — our objective-sequential forgetting is a clean counterpoint in the graph regime.

**2. ControlG.** See subsection above. Overlap: thematic (interference among ~the same objective families, incl. LP + masked reconstruction). Non-overlap: linear-probe citation benchmarks vs frozen in-context episodic eval on 8 merged Twitter graphs; no forgetting/rotation-vs-single analysis; no heuristic-floor LP; opposite headline.

**3. Pretext↔downstream alignment.** GSTBench is the sharpest: LP-pretext→LP-downstream alignment confirmed, generative robust, contrastive negative transfer, raw features competitive. GraphFM (2406.08310) and Jin et al. "Deep Insights" (2006.10141) support task-dependence; MolGraphEval and Akhondzadeh show embedding metrics ≠ downstream. **"Raw features beat every pretrained model on node *regression*" is documented nowhere** — node regression is essentially absent from graph SSL evaluation; GSTBench's "raw features are strong" is the nearest neighbor.

**4. Evaluation rigor.** Rich precedent: HeaRT (easy negatives), 2405.14985 (degree bias; **degree-matched negatives are not standard anywhere** — they're a 2024 proposal), EdgeBank (trivial baseline exposes protocol), 2306.00899 (leakage in published pipelines), 2409.20130 (rule baselines beat inductive KG-LP under sampled negatives), TLP (probing beats meta-learning), 2402.15680 (GCL protocol pitfalls). **Nobody has audited PRODIGY-style episodic label-prototype LP evaluation; no published account of a center-blind evaluator in in-context graph learning.**

**5. Collapse under masked-feature objectives.** Partially known: AUG-MAE (AAAI 2024) shows GraphMAE suffers *partial dimensional* collapse; "How Mask Matters" (2210.08344) gives theory. Our contribution is corroborating rank/norm collapse at social-graph scale inside a multi-objective pipeline — confirmatory, not novel alone.

**6. Scoop check (post-Jan-2026 flagged).** **ControlG (Feb 2026)** — main threat, framing only. **2603.10033 (late Feb/Mar 2026)** — GFM benchmark, excludes PRODIGY. **VISION (2605.24410, May 2026)** — graph few-shot ICL method, no eval critique. **OpenRFM (June 2026)** — relational-DB ICL diagnosis, adjacent. Nothing found combining multiple SSL objectives at social-media scale, and nothing auditing PRODIGY-style evaluators.

## Verdict

- **Novel:** (b) is the strongest claim — *dilution of a downstream-aligned objective under combination, even with temporal separation (rotation), plus sequential-objective catastrophic forgetting, plus the graphs-vs-objectives interleaving asymmetry*. No prior paper reports any of these; ControlG makes them timely rather than scooped. (a) is **half-known** (GSTBench: alignment + no-universal-winner; GraphFM) — our novelty is the in-context/frozen-episodic regime, heuristic floors with degree-matched negatives, transfer to a never-pretrained graph, and node regression where raw features win outright. (c) is known-ish (AUG-MAE) — supporting evidence, not a headline.
- **Evaluator forensics:** publishable angle with strong genre precedent (EdgeBank, HeaRT, 2306.00899 all became well-cited D&B/venue papers by exposing broken protocols + shipping a fix). The specific failure — **center-blind episodic LP scoring with frozen random label prototypes and degree-confounded negatives in PRODIGY-style pipelines** — is undocumented. Safest framing: pair the forensic audit with the repaired-protocol results (synergy→NM main effect inversion) rather than standalone.
- **Sharpest contrast claims:** vs **ParetoGNN/AutoSSL/WAS**: "combination-wins conclusions were reached under linear-probe evaluation on small graphs with no task-aligned heuristic floors; under frozen in-context evaluation at social-media scale, the aligned single objective dominates every combination." vs **ControlG**: "temporal separation is not sufficient — per-episode rotation *is* a temporal schedule, and it still dilutes LP by 0.077 AUC vs NM alone; scheduling cannot rescue objectives that a task doesn't need." vs **GSTBench**: "alignment holds even zero-shot on never-pretrained graphs, and extends to a task (node regression) where *no* SSL objective survives the raw-feature baseline."

Key IDs: 2602.05036, 2509.06975, 2210.02016, 2106.05470, 2403.01400, 2306.10453, 2405.14985, 2207.10128, 2306.00899, 2406.08310, 2212.05606, 2402.15680, 2206.08005, 2210.08344, 2603.10033, 2606.04320, 2605.24410, 2305.12600 (PRODIGY), 2402.07738 (Universal Link Predictor by ICL — cite for in-context LP setting).
