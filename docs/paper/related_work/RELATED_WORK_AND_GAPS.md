# Related work & gaps — literature positioning for the paper (2026-07-25)

*Produced from five parallel deep literature sweeps (multi-domain GFM/data-mixture,
transferability prediction, SSL-objective interaction, feature-vs-topology, social-media
lane), each explicitly sweeping post-Jan-2026 arXiv. Raw per-lane reports with full
citation lists: [`agent_reports/`](./agent_reports/). Companion to
[`../state_doc_jul22.md`](../state_doc_jul22.md) and the
2026-07-20 two-channels program.*

---

## 1. Verdict at a glance

| Our claim family | Status in the literature (Jul 2026) | Main threat / contrast |
|---|---|---|
| **C1. Composition is max-not-sum** (staircase, entry jumps 21/21, order-invariant post-entry level, donor/receiver matrix, rung7=rung8) | **OPEN.** Per-source attribution does not exist anywhere: no donor matrices, no leave-one-out, no source-count ladders with coverage controls. | SAMGPT (2502.05424) claims *monotone* gains 1→4 domains (toy scale); AnyGraph (2408.10700) claims scaling-law emergence; PromptGFM (2503.03313) observes multi-graph negative transfer in one table, unexplained; MDGMIX (2605.25771, May 2026) frames redundancy as compute waste. |
| **C2. Feature-divergence predicts transfer; source selection** (proxy-A ρ≈−0.92; topology weakest) | **OPEN.** No validated predictor of cross-graph SSL transfer exists; all prior evidence correlational; nobody does pre-hoc *source* selection via feature divergence. GDA mainstream argues **structure-shift-first** → our result is contrarian, not redundant. | GNNMTE (WWW 2026) plants "graph transferability estimation" flag but does *model*-hub selection; W2PGNN (2303.16458) is the structural competitor to beat; ICML 2026 graphon theory (2605.29828) argues the opposite emphasis. |
| **C3. Mechanism: SSL transfer rides on feature content; graph = feature-gathering device** (zero/noise→chance, shuffle≈no-op; yet NM beats CN/AA floors OOD) | **NOVEL, closing window.** Supervised "features carry it" canon is old (GLNN/SGC/C&S); nobody has done interventional surgery on a *pretrained transferable SSL encoder*; shuffle-vs-noise dissociation has exactly one (supervised, opposite-sign) precedent (2402.04621). | GSTBench (2509.06975) already *hypothesizes* "generative SSL wins by reconstructing rich LLM features"; ALL-IN (2605.04834, May 2026) frames feature space as THE bottleneck; LoG 2025 (2511.16767) shows structure marginal for LLM readers. Window is closing — publish the interventional version first. |
| **C4a. No universal SSL objective; alignment rules** | **HALF-KNOWN.** GSTBench shows no-universal-winner + LP-pretext→LP-downstream alignment at single-objective level. Ours adds: in-context/frozen regime, heuristic floors, never-seen-graph transfer, and node regression (absent from all graph-SSL evaluation — "raw features win regression" is documented nowhere). | GSTBench (Sept 2025) — cite as convergent, differentiate on regime. |
| **C4b. Combining objectives DILUTES the aligned one; interleaving rescues multi-graph but not multi-objective** | **NOVEL.** Every multi-pretext paper (ParetoGNN 2210.02016, AutoSSL 2106.05470, WAS 2403.01400, ControlG 2602.05036) concludes combination wins. Nobody reports dilution of the aligned objective, objective-level catastrophic forgetting, or the graphs-vs-objectives asymmetry. | ControlG (Amazon, Feb 2026) owns the "objectives interfere, schedule them" *framing* and concludes scheduling rescues combination — we are the direct counter-evidence ("rotation IS a schedule and still dilutes"). Framing scoop only; regimes don't overlap. |
| **C5. Evaluator forensics** (center-blind episodic LP, frozen random prototypes, degree-confounded negatives → synergy inverts to NM main effect) | **UNDOCUMENTED failure class** with strong genre precedent (EdgeBank 2207.10128, HeaRT 2306.10453, target-link leakage 2306.00899, degree bias 2405.14985). Nobody has audited PRODIGY-style episodic label-prototype evaluation. | None. Degree-matched negatives are a 2024 *proposal* (2405.14985), not standard — our repair is on the right side of that literature. |
| **C6. "No GFM for social networks"** (intro claim) | **TRUE only if worded precisely.** No public SSL-pretrained multi-graph multi-task social backbone exists. But: TwHIN (2202.05387) & LiGNN (2402.11139) are single-platform industrial backbones; BotTrans (2506.13795, ECML 2025) does supervised multi-source social transfer; MiNT (2406.10426, NeurIPS 2025) proves multi-network pretraining outside social media. | Wording fix required (see §3.6). |

**Bottom line:** the composition law (C1) + predictor/interventions (C2) package is the most
defensible headline — that territory is genuinely unoccupied and we hold a completed,
order-robust ladder with p≈5e-7 entry statistics. C3 is the mechanism section under it.
C4b is a self-contained secondary result (timely counter-evidence to ControlG). C5 is a
section or a spin-off D&B/companion. C6 is the venue vehicle, not a contribution.

---

## 2. Scoop watchlist (dated, closest first)

| Paper | Date | What it takes | What it leaves us |
|---|---|---|---|
| ControlG (2602.05036, Amazon) | **Feb 2026** | "Graph SSL objectives interfere; temporal allocation fixes it" framing | Opposite conclusion, no in-context/frozen eval, no heuristic floors, no forgetting analysis, no social graphs. We're the boundary condition. |
| GNNMTE (WWW 2026, no arXiv found; ACM 10.1145/3774904.3792172) | **~Apr 2026** | "Transferability estimation for GNNs" flag | Model-hub selection given a target, not source-corpus selection; discriminant score, not interpretable divergence; no mechanism. Re-verify from PDF when accessible. |
| GSTBench (2509.06975) | Sept 2025 | No-universal-objective; "SSL rides on LLM features" hypothesis | Single source graph, no combinations, no regression, no floors, observational only. |
| Graphon transfer theory (2605.29828, ICML 2026) | **May 2026** | Structure-first theory of GFM transfer | Assumes structure-informative regime; our TAG data empirically violates it — engage, don't ignore. |
| PromptGFM (2503.03313) | Mar 2025 | Observes "multiple graphs → negative transfer" | One table, no attribution, no characterization. |
| MDGMIX (2605.25771) | **May 2026** | Multi-domain redundancy observation | Efficiency framing; no donor structure, ladder, or order analysis. |
| ALL-IN (2605.04834) / GFM benchmark (2603.10033) | **May / Mar 2026** | Feature-space-as-bottleneck zeitgeist; GFM eval protocols | Neither is interventional; 2603.10033 excludes PRODIGY-style models and social data. |
| MiNT (2406.10426) | NeurIPS 2025 | Multi-network pretraining scaling ("more networks → better") | Token-transaction networks, temporal LP only; cite as paradigm proof, differentiate on domain + attribution + max-law. |

Recommendation implied by dates: **the mechanism (C3) and objective-dilution (C4b) windows
are actively closing** (3 convergent papers in Feb–Jun 2026). An arXiv timestamp for the
full package by early autumn matters more than venue choice.

---

## 3. Per-pillar positioning (condensed; full detail in agent_reports/)

### 3.1 Composition (C1)
- **Occupied:** existence of cross-domain negative transfer (GCOPE 2402.09834 motivation,
  Subgraph Pooling 2402.08907, PromptGFM); "more data ≠ better" (APT 2311.01038 "curse of
  big data", no-scaling-laws-for-graph-SSL 2408.11243); specialist-beats-generalist as
  architecture (One-Model-One-Graph 2412.00315, GP2F 2602.11629).
- **Open (ours):** functional characterization (per-target MAX + quantified dilution tax);
  ladder staircase + entry-jump statistics + post-entry order-invariance; donor/receiver
  matrix; doing it in the in-context regime where the data question was never asked.
- **Contrast claims to write:**
  (1) vs SAMGPT/AnyGraph: "apparent corpus gains are coverage effects — performance ≈ max
  over components; curves flatten once a target's best donor is in the mix."
  (2) vs GCOPE/GFT/MDGPT: "negative transfer isn't a pathology to engineer away; composition
  is lawfully max-like, so a new source's marginal value is predictable (≈0 unless it's a
  better donor)."
  (3) vs LLM mixing laws (2403.16952, 2507.09404, RegMix 2407.01492, DoReMi 2305.10429):
  "smooth-in-proportions laws don't describe graph corpora; the right analogy is the curse
  of multilinguality (1911.02116, 2010.03017, 2311.09205: mixture ≈ best component − tax)."
- Avoid the word "law" (n=8); "composition rule" / "coverage principle".

### 3.2 Predictor + interventions (C2)
- **Competitor predictors we must run to be credible:** EGI ego-graph/Laplacian score
  (2009.05204), W2PGNN graphon feasibility (2303.16458), Tree Mover's Distance (2210.01906),
  FGW / WL-distance (2202.02495), LogME/LEEP/H-score on frozen embeddings (unported to
  graphs — running them is itself a novel comparison), plus cheap descriptors
  (degree-KS, homophily deltas, spectral distance) as the topology foil. GNNMTE if
  reproducible.
- **Interventional dose-response has no precedent on real graphs** — feature rotation /
  interpolation / subspace surgery vs configuration-model rewiring. This is the causal
  differentiator; correlational-only will be attacked via the structure-first GDA canon
  (StruRW 2306.03221, Pair-Align 2403.01092, GDABench 2407.11052, 2605.29828).
- The survey 2503.09363 confirms the gap in writing ("key factors influencing
  transferability" unidentified) — quote it in the intro.

### 3.3 Mechanism (C3)
Controls reviewers will demand (from the feature-vs-topology sweep — these are the
experiments, see §4):
1. nhop>1 replication of feature-forcing (our nhop=1 made topology unusable by
   construction — the single most attackable point, and we already know it).
2. Untrained-encoder baseline, same features (2509.01541, 2311.02687) — else "it's just
   architecture".
3. SGC/NAFS feature-propagation floor — if K-step untrained smoothing matches NM transfer,
   the claim *strengthens* to "graph = feature gatherer"; if NM > SGC, quantify residual.
4. Degree-preserving (configuration-model) rewiring alongside random rewiring.
5. Class-conditional shuffle in addition to global shuffle + explicit engagement with
   2402.04621 (their within-class shuffle *helps*, supervised — explain the difference).
6. Structural-featurization arm (degree/LapPE/RWSE inputs) — show topology isn't merely
   unreadable from bio embeddings.
7. HeaRT-style personalized hard negatives + degree-corrected negatives in LP eval.
8. Per-graph homophily/feature-label alignment stats (2304.14274) — preempt "your Twitter
   graphs are trivially feature-homophilous".
- **Sell as "interventional mechanism for SSL transfer", never as "features matter"**
  (reviewers will call that known since GLNN/SGC).

### 3.4 Objectives (C4) + forensics (C5)
- vs ParetoGNN/AutoSSL/WAS: "combination-wins was concluded under linear probes on small
  graphs without task-aligned floors; under frozen in-context eval at social-media scale,
  the aligned single objective dominates every combination."
- vs ControlG: "temporal separation is not sufficient — per-episode rotation *is* a
  temporal schedule and still dilutes LP by 0.077 AUC; scheduling cannot rescue objectives
  the task doesn't need."
- vs GSTBench: "alignment holds zero-shot on never-pretrained graphs and extends to node
  regression, where no SSL objective survives the raw-feature baseline."
- FP encoder collapse: confirmatory of AUG-MAE (AAAI 2024) partial dimensional collapse —
  supporting evidence, not a headline.
- Forensics: pair the audit with the repaired-protocol inversion (synergy → NM main
  effect); genre precedent EdgeBank/HeaRT/2306.00899 all became well-cited by exposing
  broken protocols + shipping the fix.

### 3.5 Social lane (C6) + baselines
- **Hardened intro wording:** "No graph foundation model pretrained across multiple social
  networks exists: industrial backbones (TwHIN, LiGNN) pretrain on a single platform for
  recommendation; academic user-representation frameworks (Social-LLM 2401.00893, SoMeR
  2405.05275) train per corpus; generic GFMs (AnyGraph, OpenGraph) contain no event-scale
  social graphs or social tasks." Do NOT write "nobody has shown gains from multiple social
  graphs" unqualified (BotTrans falsifies it for supervised DA).
- **Baselines reviewers will demand:** supervised TwiBot-20 skylines in-table (BotRGCN
  F1≈.87, RGT 86.5/87.9, SeBot 2405.11225, LMBot 2306.17408); SSL precedents SATAR
  (2106.13089) + BotSSCL; Social-LLM-style bio-embedding+shallow-graph baseline; BotArtist
  (2306.00037) as the existing cross-dataset transfer table (features-only, no graph);
  suspension prior 2306.03502 (same Ukraine-war data lineage — check overlap and cite);
  ideology priors Barberá 2015 + Retweet-BERT (2207.08459).
- **Protocol bridge:** our frozen few-shot numbers will be read as non-comparable; frame as
  label-efficiency/frozen-transfer, print the supervised skyline in the same table, add a
  same-protocol supervised(-ish) baseline.

### 3.6 Venues (deadlines as verified 2026-07-25 by the sweep)
| Venue | Deadline | Fit |
|---|---|---|
| WSDM 2027 (Hong Kong) | **abstract Aug 17, paper Aug 24, 2026** (verified) | Best fit per publication record of closest work — but 4 weeks out: only viable with existing results + P0 rigor + analysis-only experiments. Risky. |
| ICWSM 2027 (Edinburgh) | rounds **Sep 15, 2026** / Jan 15, 2027 (verified) | CSS framing ("transfer structure between events"); Sep 15 is 7.5 weeks out — plausible for the composition+donor story. |
| WWW 2027 | typically abstracts ~Oct (UNVERIFIED — check) | Best full-package fit (Luceri WWW'24, cross-platform WWW'26 precedents + graph track). |
| NeurIPS 2027 D&B (or 2026 if any late track) | typically ~May | Corpus + 8×8 transfer matrix + repaired evaluator as a benchmark release (TwiBot-22/MiNT precedents). |
| TMLR | rolling | Fallback for the ML story with no deadline pressure. |
| arXiv | ASAP | The real deadline given §2 — timestamp before the C3/C4b windows close. |

---

## 4. Gaps → experiments (prioritized)

Costs assume Tucker GPUs 0–2, ~80 min per 40k-step rung, evals ≈ minutes per arm,
`--neighbor_sampling_source_subset` for arbitrary sub-mixtures from the all8 artifact.

### P0 — rigor debt; required for ANY submission
| ID | Experiment | Fills | Cost |
|---|---|---|---|
| **E1** | **Seeds + noise floor.** 3–4 pretrain seeds on pivotal arms (2 specialist donors, all8, NM vs MIX rotation); fix shared-loader eval reseed (eval-episode seeding is per-split, not --seed); publish MDE table; pre-demote sub-MDE effects to equivalence claims. | Every 1-seed number we currently quote; reviewer-proofing all pillars | ~12–16 rungs ≈ 1–2 GPU-days |
| **E2** | **Floors in every table.** Untrained (random-init) encoder with identical features; raw-feature probe (exists for regression — make universal); SGC/NAFS-style K-step untrained smoothing floor; supervised TwiBot-20 skylines printed alongside. | C3 controls #2–3; C6 protocol bridge; 2509.01541/2311.02687 attack | Eval-only + tiny CPU/GPU |
| **E3** | **LP eval hardening.** Add HeaRT-style personalized hard negatives to the repaired pair evaluator (config change); keep degree-matched as headline; leakage audit write-up. | C5 completeness; 2306.10453/2405.14985 alignment | Hours |

### P1 — headline-hardening; each fills a named literature gap
| ID | Experiment | Fills | Cost |
|---|---|---|---|
| **E4** | **Volume-vs-coverage decomposition.** (a) Within-source data-scaling curve (subsample best donor at 10/25/50/100%, fixed steps); (b) matched-episode mixture comparison (1 graph full vs k graphs at same total episodes — the ladder is compute-matched already; make the *data-budget* axis explicit). | The one confound in C1 ("you just added data"); positions against 2408.11243 + mixing-law lit | ~6–8 rungs ≈ 1 GPU-day |
| **E5** | **Functional-form comparison (analysis-only).** Fit mixture ≈ max(components) vs additive vs smooth-in-proportions (LM mixing-law forms) on the EXISTING ladder + pairs + order-robustness data; model-selection table (AIC/loo). | Turns staircase into a law-vs-law result; the sharpest C1 deliverable; zero GPU | Days of analysis |
| **E6** | **Predictor-at-scale.** All 64 ordered pairs (matrix exists): our divergence axes vs EGI, W2PGNN, TMD, FGW/WL, LogME-on-embeddings, degree-KS, homophily-gap; hierarchical/QAP permutation stats; missing-bio-rate as confound control. | C2 vs GNNMTE/W2PGNN; the "no validated metric" gap in 2503.09363 | CPU-heavy + small GPU; ~week incl. baselines |
| **E7** | **Interventional dose-response.** On 2 mid-size graphs: feature-cloud rotation / interpolation toward target / subspace removal (dose-graded), vs configuration-model rewiring at matched degree sequence; manipulation checks at every rung; transfer as monotone function of dose. | The causal differentiator for C2/C3; literally unprecedented | ~10–15 mid-size rungs ≈ 1–2 GPU-days |
| **E8** | **Source-selection payoff.** Pre-register (repo-hash) predicted-best-k mixture per target from feature divergence; train the k-source subsets via subset knob; show ≈ all-8 at k=2–3 and ≫ worst-k. | C2's applied payoff; "which graph should I pretrain on" — nobody answers this | ~8 rungs ≈ 1 GPU-day |
| **E9** | **nhop=2 feature-forcing rerun** + structural-featurization arm (degree/RWSE inputs). | C3 control #1 and #6 — our own known weakest point | ~4–6 rungs |
| **E10** | **Objective 2×2 completion.** Joint-loss (simultaneous gradients) vs rotation (schedule) vs singles, with E1 seeds → completes "neither mixing NOR scheduling rescues" against ControlG; add conditional-shuffle arm from C3 control #5. | C4b vs ControlG's central claim | ~6 rungs |

### P2 — scope extensions; pick per venue
| ID | Experiment | Fills | Cost |
|---|---|---|---|
| **E11** | **One alien donor** (e.g., ogbn-arxiv as 9th source): does max-composition hold with a truly out-of-family donor? | External validity of C1; connects to generic-GFM lit; preempts "all your graphs are Twitter" | Data-eng heavy; ~week |
| **E12** | **Temporal LP** as a NEW measurement (repaired evaluator supports via config; fix the ~0.15% pair-disjointness leak first). | New task family; C4a breadth | Days |
| **E13** | **PRODIGY-repo audit:** check whether the original public PRODIGY zero-shot LP path shares the center-blind/frozen-prototype defect. If yes, C5 generalizes beyond our fork → forensics section gains teeth (possibly its own D&B paper with the corpus release). | C5 impact | Code reading + small reruns |
| **E14** | **Capacity control.** One ~10× model on all8: if max-composition persists at higher capacity, the "limited capacity forces interpolation" alternative explanation dies; also the GFM-claim scale armor from the state doc. | Strongest scientific objection to C1; state-doc scaling TODO | Multi-day run |

**Suggested sequencing** (compute-serial, analysis-parallel):
E5 + E6 start now (no GPU); E1–E3 immediately (they gate everything); then E4 → E7 → E8
(composition/causal spine), E9–E10 alongside; E11–E14 only after venue choice.

---

## 5. Recommended paper shape

**One main ML paper** — working title direction: *"Coverage, not accumulation: what
multi-graph self-supervised pretraining actually transfers"* —
1. Composition rule: staircase + entry-jump stats + order-invariance + donor matrix +
   functional-form fit (E5), volume control (E4).
2. Predictor: feature divergence at scale vs structural competitors (E6) + interventional
   dose-response (E7) + selection payoff (E8).
3. Mechanism: ablation surgery with the P0/P1 controls; NM's above-floor OOD adjacency
   signal as "relational signal lives in feature space".
4. Secondary: objective alignment + dilution (E10) — or split out if space demands.

**Optional companion (D&B / ICWSM):** corpus + 8×8 transfer benchmark + evaluator
forensics (E13) + protocol bridge to supervised skylines.

Master must-cite list and per-lane detail: see [`agent_reports/`](./agent_reports/).
