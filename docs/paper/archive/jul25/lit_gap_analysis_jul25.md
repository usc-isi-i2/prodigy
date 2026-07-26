# Literature gap analysis — 2026-07-25

**What this is.** A five-thread literature sweep (GFM landscape; data mixtures / negative
transfer; SSL objectives / multi-task; features-vs-topology / shortcuts; transferability
prediction / graph DA) run against our *current, post-sLP-rescore* findings, to answer:
which of our claims are novel, who do we argue against, what is missing for a
NeurIPS/ICLR/ICML submission, and what should we add. Every citation below was
web-verified on 2026-07-25 (arXiv/venue pages); the four most load-bearing
single-sourced items (MiNT, FAF, 2511.16767, 2508.20583) were independently re-fetched.

**Deadlines (verified).** ICLR 2027: abstract **Sept 19**, paper **Sept 24, 2026** (~8
weeks). ICML 2027: unannounced, historically late January (~6 months). NeurIPS 2026
passed (May 2026).

---

## 0. Verdict

Our two strongest claims sit in lanes that are **genuinely unoccupied but rapidly
crowding**:

1. **Mixture = max, not sum** (ladder staircase + never-beats-best-specialist +
   no-OOD-bonus): *no published paper states this rule, runs the best-single-donor
   control at matched compute, or fits any functional form to graph-pretraining corpus
   composition.* The multi-domain-GFM literature (GCOPE, SAMGPT/MDGPT, OFA, MiNT,
   GraphFM) claims merging helps while systematically omitting exactly the control we
   run. A 2025–26 deflationary wave (GraphLand, Fair-GFM-eval, OMOG, MDGCL, APT,
   GraphSculptor, MDGMIX) is converging on "merging is problematic" *without* the
   per-donor attribution, the max rule, or a predictor. This is the sharpest, most
   time-sensitive novelty we own.

2. **Transfer is predictable from the feature channel** (feature-cloud divergence
   ρ≈−0.9; topology divergence ~0; noise-fatal/shuffle-harmless surgery): *no validated
   pre-hoc predictor of cross-graph pretraining transfer exists* (the 2025
   transferability survey names the gap; W2PGNN is topology-only and explicitly assumes
   attribute spaces don't overlap — the premise our frozen-text-embedding regime
   overturns). Nobody has regressed a measured n×n transfer matrix on competing
   interpretable divergences, decomposed donor/receiver asymmetry, or moved feature
   divergence *interventionally* on real graphs. The structure-first camp (GFT,
   RiemannGFM, GFSE, graph-vocabulary position paper, and the ICML 2026 graphon theory
   2605.29828) is the explicit foil.

3. **Objective dilution** (every NM-containing mixture < NM alone at matched compute;
   no universal objective; FP collapse; the evaluator autopsy): this **contradicts the
   published record** (AutoSSL, ParetoGNN, ControlG all claim mixtures/scheduling win —
   none runs matched-compute best-single-per-task controls). A complete 2^k−1 subset
   lattice at matched compute is unpublished. High-value but needs mechanism + one
   replication to survive review; best as a second act or a second paper.

The biggest external-validity hole in everything above: **all 8 corpora are Twitter
graphs with one feature extractor and one backbone (PRODIGY)**. The within-modality
design is also our best control (same schema, same features ⇒ the "cross-domain
semantics" excuse for negative transfer dies), but reviewers will demand at least one
out-of-modality or second-backbone replication.

---

## 1. Novelty audit — claim by claim

| Our claim | Status in literature | Key prior work | Our edge |
|---|---|---|---|
| Mixture ≈ max over in-mix donors (staircase; jump when your graph enters) | **Not shown anywhere.** Additive-vs-max question unposed | OFA ind-vs-joint table (unanalyzed small deltas); GraphFM "competitive with specialists"; APT "curse of big data"; Cole CVPR'22 (vision SSL: extra domain buys nothing); XLS-R vs monolingual; curse of multilinguality (mixture < max at fixed capacity) | First statement + staircase dynamics + order-robustness + matched-compute specialist control |
| Merged never beats best specialist in-domain; no OOD bonus | Partial fragments, never as the control | GCOPE/SAMGPT/MDGPT *deleted* the specialist control (re-run singles on merged data); MiNT compares to target-side specialists only | We run the donor-side control; recast Δ(mix − best donor) as required GFM reporting |
| 8×8 asymmetric donor/receiver matrix (universal donors, isolates) | **No measured n×n matrix for graph SSL.** Template exists in CV/NLP | Taskonomy CVPR'18; Vu EMNLP'20; LangRank ACL'19; on graphs only pairwise/1×n (EGI, GSTBench, Frasca wksp'24) | First graph instance + donor/receiver effect decomposition |
| Feature divergence predicts transfer (ρ≈−0.9), topology doesn't | **No validated pre-hoc predictor for graphs**; survey names gap | W2PGNN KDD'23 (graphon/topology-only feasibility, dismisses attributes); EGI (structural bound); GNNMT WWW'26 (post-hoc, needs checkpoints); 2503.09363 survey | Predictor race (feature PAD vs degree-KS vs graphon vs LogME/LEEP), LOGO-validated, on a measured matrix |
| Within-source sampling beats cross-source (source-discrimination shortcut) | **Not shown in any merged-corpus SSL** | Robinson NeurIPS'21 (shortcut theory, single corpus); Liu & He ICLR'25 (dataset origin trivially classifiable); GCC KDD'20 *uses* cross-graph negatives unexamined; MDGMIX ICML'26 *adds* domain discrimination as a feature | Causal knob + fix; directly argues against MDGMIX-style designs |
| NM relies on feature content (noise fatal / shuffle harmless / rewire harmless) | Triad **not shown**; pieces exist supervised-only | Lee ICML'24 (within-class shuffle, supervised); Ma ICLR'22, Bechler-Speicher ICML'24 (graph-substitution, supervised); GraphMAE/BGRL/GCC never ablate channels | First coordinated channel surgery on a *pretrained transfer* model |
| No universal SSL objective; NM best for LP; mixing dilutes | Premise shown; **dilution contradicts record**; lattice unpublished | AutoSSL ICLR'22 + ParetoGNN ICLR'23 (claim mixtures win; no matched-compute control); ControlG 2/2026 (names interference, claims scheduling rescues); GSTBench CIKM'25 (1-source: most SSL fails to transfer, GraphMAE best); Sun NeurIPS'22 (pretraining ≈ no gain, molecules) | 7-arm lattice at matched compute + weighted-sum control + "alignment not mixing" framing |
| FP representation collapse (536/4568 directions) | Partially (vision theory; graph fix papers) | U-MAE NeurIPS'22 (MAE dimensional collapse, vision); AUG-MAE AAAI'24 (GraphMAE uniformity deficient) | First in-the-wild direction-count collapse in multi-graph pretraining; counterexample to GSTBench's "GraphMAE transfers best" |
| Evaluator artifact: "emergent synergy" dissolved on fix | Genre exists; **no such case study** | HeaRT NeurIPS'23; Aiyappa NeurIPS'24 (degree-null near-optimal); Sun ACL'20 (KGC eval artifact); Platonov ICLR'23 (leakage); Fair-GFM-eval 6/2026 | First episodic/in-context LP autopsy + checklist (endpoint encoding, prototypes, degree-matched negatives, permutation gates) |
| Raw features beat pretrained embeddings on regression | Partial analogs, not for TAG-GFMs | Sun NeurIPS'22 (molecules); GraphLand (LightGBM strong); Fair-GFM-eval | First in few-shot in-context TAG regime; needs a GBDT arm to be bulletproof |

---

## 2. Positioning map — who we argue against, who supports us

**We contradict (must engage head-on):**
- *Merge camp:* OFA (ICLR'24), GCOPE (KDD'24), SAMGPT (WWW'25)/MDGPT, UniGraph (KDD'25),
  AnyGraph, **MiNT (NeurIPS'25 D&B)** — the cleanest counter-claim ("more networks →
  better zero-shot", 84 Ethereum token networks). Defusal: different modality/task;
  their baseline is *target-side* specialists, not best in-mix donor; no matched
  compute; no donor attribution. GraphFM's fine print ("competitive with specialists")
  actually concedes our point.
- *Structure-first camp:* graph-vocabulary position paper (ICML'24), GFT (NeurIPS'24),
  RiemannGFM (WWW'25), GFSE (ICML'25), W2PGNN (KDD'23), **Zhu et al. "When Do GFMs
  Transfer? A Data-Centric Theory" (ICML 2026, 2605.29828)** — graphon/structural
  discrepancy theory marketing data-curation guidance. Our feature-channel dominance is
  the direct empirical stress test of all of these.
- *Mixture-optimists:* AutoSSL (ICLR'22), ParetoGNN (ICLR'23), **ControlG (2602.05036)**
  — all claim combining/scheduling SSL objectives wins; none compares against the best
  single objective per task at matched budget. Our lattice is the missing control.

**We extend / stand on:**
- *Deflationary wave:* Hu ICLR'20 (negative transfer), APT NeurIPS'23 ("curse of big
  data"), GraphLand (GFMs fail on social/industrial), Fair-GFM-eval (ICML'26 wksp),
  GSTBench (CIKM'25), Ma et al. 2408.11243 (no scaling laws in graph SSL), OMOG
  (per-graph experts because merging fails), Subgraph Pooling IJCAI'24 (negative
  transfer between similar graphs), UniAug (concedes heterogeneity blocks data scaling).
- *Feature-dominance neighbors:* GLNN ICLR'22, C&S ICLR'21, SGC, **FAF (ICML 2026,
  2601.19449** — fixed mean-aggregation features + MLP rival GNNs 12/14; adopt as
  methodology), GIANT ICLR'22, "When Structure Doesn't Help" (2511.16767), "A Graph
  Talks, But Who's Listening?" (ACL-F 2026) — all supervised or LLM-side; none touch
  pretrained GNN GFMs, causal surgery, or transfer laws.
- *Eval-hygiene genre:* Shchur'18, HeaRT, Aiyappa'24, Platonov'23, Sun ACL'20.
- *CSS bridge:* LOBO ACSAC'18, TwiBot-22, Hays WWW'23 (best paper), Gabriel Sci Rep'23,
  IOHunter AAAI'25, Verhoeven CL'25 — cross-event failure is documented; nobody connects
  it to pretraining-corpus choice. Our donor-selection rule is that connection.

**LLM-side framing vocabulary (cite, don't compete):** DoReMi NeurIPS'23, Data Mixing
Laws ICLR'25, RegMix ICLR'25, Aioli ICLR'25 (notably: nothing beats stratified
sampling), BiMix, D-CPT; curse of multilinguality (Conneau ACL'20, Wang EMNLP'20,
X-MOD NAACL'22); XLS-R vs monolingual (matched capacity); model soups ICML'22 / task
arithmetic ICLR'23 (weight-space merging can beat max — clean contrast to our
data-space max).

---

## 3. Scoop watch (ranked)

| Risk | What | Why it collides | Defusal / urgency |
|---|---|---|---|
| **High** | Zhu et al. ICML 2026 (2605.29828) group | Theory paper explicitly promises "data curation guidance for GFM transfer"; an empirical companion = our matrix+predictor | Their discrepancy is structural; have the feature-vs-topology race ready. Move fast |
| **High** | ControlG (Amazon) | Same interference problem, opposite conclusion; a follow-up adding best-single controls would collide with the dilution claim | Cite + differentiate on matched compute and the missing control |
| Med | MDGMIX (ICML'26) | Multi-domain redundancy + domain-discrimination-as-feature | Opposite mechanism sign: we show domain discrimination is the *failure mode* |
| Med | GSTBench group (CIKM'25) | 1×n → n×n extension is their obvious next paper | Our matrix + mixtures + floors already deeper |
| Med | ICLR'26 sub 6wNx3KpS3d | CSBM dose-response of feature/structure shift on SSL transfer | Synthetic-only; our real-graph interventions outrank it — but the headline overlaps. Track acceptance |
| Med | MiNT | "More networks better" narrative at NeurIPS D&B | Engage explicitly (see §2) |
| Low-med | IOHunter group (AAAI'25) | Same substrate (GNN+LM embeddings, influence ops); a cross-campaign transfer matrix from them lands on our CSS bridge | Our 8-graph corpus + tasks is deeper; consider citing as the GFM-for-social anchor |
| Low | "When Structure Doesn't Help" / "A Graph Talks" | Feature-dominance flank (LLM-side) | Neither touches pretrained GFMs/transfer; cite as convergent |

Venue note: **ICML 2026 now has a dedicated GFM workshop** — the deflationary threads
are consolidating there; also a fallback venue for a fast workshop version.

---

## 4. Gaps → what to add (prioritized)

### P0 — hardening that blocks every claim (start immediately)
1. **Multi-seed + MDE table.** Every headline is ~1 seed. Use the shared-loader reseed
   (eval episode seed is per-split!) + cross-dataset agreement; publish a
   minimal-detectable-effect table and pre-demote sub-MDE effects to equivalence claims.
2. **Matched-compute bookkeeping.** Formalize Δ(mix − best in-mix donor) per target at
   equal episode budget; this is the "missing control, weaponized" figure.
3. **Temporal-LP rescore.** Same defect as sLP; nothing LP-adjacent ships before this.

### P1 — free analyses on existing artifacts (this week; do regardless of venue)
4. **Mixture-law functional-form race.** Fit max vs additive vs log-linear (LLM
   mixing-law forms; cite Ye/BiMix/RegMix) on ladder + order-robustness data; report
   staircase + per-rung regret vs oracle donor. Turns the ladder into a law-shaped
   figure without new training.
5. **Predictor race on the 8×8.** Feature-PAD vs degree-KS/clustering/spectral vs
   graphon-overlap (W2PGNN-style) vs LogME/LEEP (post-hoc baselines), LOGO-validated,
   regret-vs-oracle; add donor/receiver effect decomposition (row/col effects,
   asymmetry index). Predictor-error audit already exists; add competitors + stats
   (hierarchical/QAP for dyadic non-independence).
6. **Dilution mechanism diagnostics.** Per-objective gradient cosines across training +
   effective-rank/uniformity curves per arm (quantifies FP collapse: 536/4568
   directions → effective-rank curve). Analysis passes over existing checkpoints;
   positions against U-MAE/AUG-MAE and explains ControlG's "Disagreement" regime.

### P2 — cheap eval-time experiments (days each)
7. **Surgery triad at LP eval.** Noise/shuffle/rewire on NM arms including the
   never-trained graph (twibot20 +0.109 over heuristics): settles "is NM's adjacency
   signal feature-borne" — the single question both papers hinge on. (This re-points the
   old mix_slp_ablation at NM.)
8. **Scaffold-vs-signal grid.** {true graph, degree-preserving rewire, feature-kNN
   graph, no graph} × {learned message passing, fixed-mean aggregation (FAF-style)} on
   frozen features, classification + LP. "kNN ≈ true ≫ rewire" certifies
   topology-as-scaffold; cleanly resolves our "neighbor features help a ton" caveat
   with a published methodology anchor (FAF, ICML'26).
9. **GBDT floor.** LightGBM on raw features (+ neighbor-aggregated features,
   GraphLand-style) for regression + classification. Cheap; bulletproofs "raw features
   win" and pre-empts the strongest known baseline family.
10. **n_hop sensitivity.** Eval-time n_hop=2 on existing checkpoints + one NM pretrain
    at n_hop=2 on a mid-size graph. Kills the self-identified ill-posedness of the
    "force topology" experiment (1-hop scope mechanically caps visible structure —
    shaDow-GNN is the support cite; no published hop ablation exists for episodic GFMs,
    so even a small one is novel).

### P3 — new training, scoped (1–2 weeks GPU; decide by paper choice)
11. **Interventional dose-response on feature divergence** (Paper A's causal
    centerpiece). Embedding rotation / interpolation / on-manifold subsampling on
    mid-size graphs with topology fixed; degree-preserving rewire as null; natural
    temporal within-event slices plotted on the same curve. Nobody has done this on
    real graphs (closest is synthetic CSBM). Upgrades ρ≈−0.9 from correlation to
    causation — the difference between "observed" and "explained".
12. **Predicted top-k mixture at matched budget** (Paper A's payoff). Many k-subsets
    already exist from ladder + order-robustness; the new content is *predicted-best*
    selection vs random/size/language-selected controls ≈ all-8. Few new pretrains.
13. **Capacity sweep on one mixture.** 2–3 model sizes: does max become sum with
    capacity? Separates capacity-dilution (multilinguality story) from shortcut;
    whichever way it lands it's a figure.

### P4 — generality (venue-deciding; the big fork)
14. **Second backbone** (GraphMAE-class or BGRL on our corpora) **or second modality
    corpus** (e.g., 3–4 TSGFM citation graphs through our pipeline). Tests whether
    mixture=max and feature-channel dominance are PRODIGY/Twitter artifacts; directly
    engages GSTBench ("GraphMAE transfers best" — our FP-collapse result predicts
    otherwise on our corpora). This is the main determinant of top-venue ceiling.
15. **(Optional) LLM-on-bios baseline row** for labeled tasks — defuses "why GNNs in
    2026" and sharpens the text-models-in-disguise frame.

---

## 5. Packaging recommendation

**Paper A (primary; target ICLR 2027, Sept 24).**
*"Transfer in multi-source graph pretraining is selection, not addition."*
Spine: ladder/staircase + order robustness (mixture=max) → 8×8 matrix + asymmetry →
predictor race (feature channel wins, LOGO) → interventional dose-response (causation)
→ within-source sampling shortcut (mechanism) → predicted top-k selection ≈ all-8
(payoff) → no-OOD negative result. Needs: P0, P1.4–5, P2.7–9, P3.11–12; P4.14 if it
fits. ~80% of training exists; the 8-week risk is analysis+writing, not compute.
Fallbacks if hardening slips: ICML 2027 (Jan) with P4 included, or the ICML GFM
workshop as a fast flag-plant.

**Paper B (second act of A, or standalone for ICML 2027).**
*"Alignment, not mixing: what SSL objectives transfer on graphs."*
Lattice at matched compute + NM main effect + dilution + FP collapse + evaluator
autopsy/checklist + gradient-conflict mechanism. Contradicts AutoSSL/ParetoGNN/ControlG
record — high reward, but needs P1.6, P2.10, P0.3, and ideally P4.14 to survive "one
model family" review. The evaluator autopsy could also be a strong workshop/D&B
methods contribution on its own.

**Paper C (spin-off, optional).** The 8-graph social-GFM testbed + transfer atlas +
divergence pipeline + evaluator checklist as a Datasets & Benchmarks / ICWSM artifact
(no GFM pretraining corpus on retweet graphs exists; GraphLand has social eval only).

**Open forks for Philipp:** (a) A alone vs A+B merged for ICLR (recommend: A with B's
NM/dilution as one section, full B later); (b) invest P4.14 second backbone now vs
scope title to "episodic/PRODIGY-class" (recommend: start the GraphMAE-class arm now —
it de-risks both papers and GSTBench gives it a ready-made foil); (c) synthetic CSBM
add-on for the feature-channel story (cheap insurance against "correlational" reviews
if P3.11 slips).
