# GFM Retweet-Graph Program — Consolidated Findings

**Can one pretrained encoder give transferable node representations across many
retweet graphs and many downstream tasks — and does merging sources or engineering
the SSL objective get us there?** Across ~13 experiments the answer is a qualified
*mostly no, with two sharp exceptions*: today's Neighbor-Matching (NM) pretext is a
**feature-content** learner (topology ≈ chance); **merging sources** matches or beats
single-source transfer but never adds an out-of-distribution bonus and pays a small
in-domain tax; and the **only** way we have produced an encoder strong on *both*
feature and topological tasks is **rotating heterogeneous SSL pretexts (NM⊕CL⊕FP)** —
a hand-engineered multi-head objective (E4) does *not* do it.

_Status: umbrella synthesis over the per-experiment write-ups (each linked below).
**Every headline is 1 seed** — the program has not yet run a seed sweep. Numbers are
lifted verbatim from each thrust's own `FINDINGS.md`/`RESULTS.md`; absolute values are
only comparable **within** an experiment (corpus / sampling / checkpoint differ across
them — see §5.6). Last consolidated 2026-07-20._

> **⚠️ Partially superseded (2026-07-23), not yet rewritten.** Every
> static-link-prediction number in this document — including the headline "MIX is the
> only generalist / LP is a 3-way synergy" and the 0.42 → 0.32 → 0.76 non-monotonic
> lattice — came from an evaluator later found invalid. On the rescore of the same
> frozen checkpoints, **link prediction is a neighbor-matching main effect that
> rotation dilutes**: NM is the best arm on all 5 datasets and MIX sits near the
> heuristic floors. There is no synergy. The classification, regression, feature-ablation
> and NM-transfer sections are unaffected.
>
> Corrected read: [`../multitask_ssl/FINDINGS.md`](../../objectives/multitask/multitask_ssl/FINDINGS.md) ·
> defect details: [`../multitask_ssl/FINDINGS_rescore.md`](../../objectives/multitask/multitask_ssl/FINDINGS_rescore.md).
> Rewriting this synthesis against the valid numbers is still open work.

---

## 1. Executive summary

Five findings recur across the program and are each supported by ≥2 independent thrusts:

- **NM is a neighborhood-*feature-content* matcher, not a topology learner.** Destroying
  real neighborhood content collapses NM to chance exactly like deleting features, while
  scrambling the feature↔node binding is harmless (feature_ablation). Independently,
  **feature-cloud separability predicts transfer** (proxy-A-distance ρ≈−0.92) while raw
  degree-distribution distance is the weakest predictor (ρ≈−0.6) (similarity_vs_transfer).
  Two analyses, same conclusion: features carry the signal, topology rides along.

- **Merged ≥ single-source on transfer, but merging never buys OOD — and taxes small
  graphs in-domain.** The early "single beats merged" *inversion does not reproduce*
  under a fair comparison (nm_transfer_matrix, nm_covid_midterm). The all-8 merged model
  loses only ~.006–.04 AUC in-domain vs each specialist, the tax landing hardest on the
  small/topical graphs, and buys **+.09–.16** on graphs with no strong donor
  (nm_single_source_matrix, nm_ladder). No merge ever beats a single source
  *out-of-distribution*.

- **Within-source episode sampling beats naive/proportional — the cross-source shortcut
  is real.** Confining each NM episode to one source removes a source-discrimination
  shortcut and gives a small consistent gain; under size imbalance, balanced
  within-source *rescues* a starved small domain (midterm 0.31→0.43, above its own
  single-source specialist) (nm_cross_source_shortcut, nm_covid_midterm,
  sampling_strat_comparison). A cross-source-probability sweep confirms p=0 (fully
  within-source) is best.

- **⚠️ SUPERSEDED (see banner) — The "learns both feature *and* topology" encoder exists,
  but only via pretext rotation, and it's a 3-way synergy.** Rotating NM/CL/FP one-per-episode (MIX) is the
  *only* arm above chance on zero-shot static link prediction (0.76 vs ≤0.47 for every
  single objective) while staying near-best on classification (multitask_ssl_rotation).
  **No pair reproduces it** — capability is non-monotonic (singles 0.42 → pairs 0.32 →
  triple 0.76); the topological transfer is emergent only in the *complete* set
  (multitask_ssl_pairs).

- **Engineering the objective directly fails; frozen SSL transfers only when the pretext
  structurally matches the task.** The multi-head E4 (MFR⊕directed-LP⊕structural)
  *degrades* the encoder below the NM control on both axes (topology_feature_ssl).
  Frozen-probe matrix: the only objective that clears a floor is **single-source NM →
  link prediction** (+0.23 AUC, pretext ≈ task); for regression *nothing* beats raw
  features and pretraining is strictly harmful (pretrain_probe_matrix).

**Where this lands (one line):** we understand *what NM learns* (feature content) and
*how to combine sources* (within-source, balanced) and *how to get topology* (rotate
pretexts, don't hand-build a loss) — but every result is single-seed, and no single
frozen encoder is simultaneously best on classification, regression, and link
prediction. See §5.

---

## 2. Scope & methodology

### The program-wide question

We pretrain graph encoders self-supervised on **retweet graphs** (nodes = users with
768-d GTE multilingual bio embeddings; edges = retweets) and read the **frozen**
representation's transfer to downstream tasks. Every thrust is one controlled slice of:
*which pretraining data + objective + sampling gives the most transferable rep, and
what does the encoder actually use?*

### The 8 graphs (the shared column space)

`ukr_rus_twitter`, `covid19_twitter`, `midterm`, `covid_political`, `election2020`,
`ukr_rus_suspended`, `twibot20`, `cp_hk_twitter`. A **merged** graph is a disjoint
block-concat (users namespaced per source, no cross-source edges, provenance kept).
The three big/early sources (ukr, covid, midterm) form the SSL pretrain corpus; the
rest are held-out transfer targets.

### The thrust map

| # | Thrust | Folders | One-line finding |
|---|--------|---------|------------------|
| A | Graph diagnostics | [graph_divergence](../../../setup/graph_divergence/README.md), [similarity_vs_transfer](../../graphs/transfer_prediction/similarity_vs_transfer/FINDINGS.md) | Feature-cloud separability predicts transfer (ρ≈−0.92); topology distance barely does. |
| B | NM merged-vs-single (cross-source shortcut) | [nm_transfer_matrix](../../transfer/matrices/prodigy_nm/merged_vs_single/nm_transfer_matrix/RESULTS.md), [nm_cross_source_shortcut](../../transfer/ablations/prodigy_nm/episode_sampling/nm_cross_source_shortcut/RESULTS.md), [nm_covid_midterm](../../transfer/matrices/prodigy_nm/merged_vs_single/nm_covid_midterm/RESULTS.md), [sampling_strat_comparison](../../transfer/ablations/prodigy_nm/episode_sampling/sampling_strat_comparison/) | Inversion doesn't reproduce; within-source > naive; balanced rescues small domains. |
| C | NM transfer geometry (8×8 + ladder) | [nm_single_source_matrix](../../transfer/matrices/prodigy_nm/single_source/nm_single_source_matrix/FINDINGS.md), [nm_ladder](../../transfer/ladders/prodigy_nm/baseline/nm_ladder/RESULTS.md) | Specialists beat merged in-domain everywhere; covid/ukr universal donors; cp_hk an island. |
| D | Features vs topology | [feature_ablation](../../objectives/topology_vs_features/feature_ablation/FINDINGS.md) | NM uses real feature *content*; topology alone ≈ chance at n_hop=1. |
| E | SSL-objective studies | [topology_feature_ssl](../../objectives/topology_vs_features/topology_feature_ssl/FINDINGS.md), [multitask_ssl](../../objectives/multitask/multitask_ssl/FINDINGS.md) | A multi-head objective (E4) fails. ⚠️ The "MIX is the only generalist / LP is a 3-way synergy" reading is superseded — LP is an NM main effect (see banner). |
| F | Frozen-probe strategy benchmark | [pretrain_probe_matrix](../../objectives/frozen_probes/pretrain_probe_matrix/FINDINGS.md), [pretrain_strategy_benchmark](../../../setup/pretrain_strategy_benchmark/README.md), [covid_task_transfer_matrix](../../../setup/covid_task_transfer_matrix/README.md) | Only NM→LP clears a floor; raw features beat all pretraining on regression. |
| G | New downstream tasks (enablers) | [node_regression](../../../setup/node_regression/README.md), [static_link_prediction](../../../setup/static_link_prediction/README.md) | Continuous + edge-level tasks with real headroom; used as the topology/feature axes above. |
| H | New datasets (enablers) | [twibot20_transfer](../../../setup/twibot20_transfer/README.md), [cp_hk_twitter](../../../setup/cp_hk_twitter/README.md), [cp_hk_transfer_in](../../../setup/cp_hk_transfer_in/README.md) | twibot20 = easy transfer target (~.92 zero-shot); cp_hk = the isolated island. |

### How to read this document

- **The joint bar is `min(feature_score, topological_score)`, never the mean.** An
  encoder excellent at one family and useless at another is not a general encoder.
  Feature axis = node-classification ROC-AUC (and node-regression Spearman ρ as a
  secondary, noisy axis); topological axis = 0-shot static-LP ROC-AUC.
- **Floors matter.** "Improves" means *beats a floor* (`raw_feat` ridge with no GNN;
  `random_init` untrained encoder; `raw_degree` leakage control), not *closest to a
  saturated ceiling*.
- **Matched-40k, 1 seed** unless noted. Treat sub-.02 AUC gaps as noise.

---

## 3. Experiment setup (shared)

- **Encoder (default):** 768-d GTE bio embeddings → mean-aggregating GraphSAGE,
  1 layer / 1 hop, emb_dim 256, undirected message passing. Structural-input and
  aggregation variants live in thrust E.
- **NM eval regime:** 30-way / 3-shot (always ≥3 shots — 0-shot NM has no support
  prototypes and collapses to chance, an eval artifact that once made every model look
  random). ROC-AUC is near-ceiling (~.9+) on retweet graphs, so **accuracy is the
  discriminative DV** for NM transfer; AUC is reported for LP/classification.
- **Frozen-probe eval tasks:** node classification (ROC-AUC, 10-shot), node regression
  (Spearman ρ, 10-shot, log1p on the 6-target profile panel), static link prediction
  (ROC-AUC, 0-shot; ~85/15 edge split with held-out edges removed from the background).
- **Checkpoint hygiene:** the trainer saves no ckpt at the final 0-indexed step, so a
  "40k budget" lands `state_dict_40000` only with `epochs:5`; the rotation runs are read
  at their true-highest `state_dict_30000`. Numbers across thrusts use each experiment's
  stated checkpoint.
- **Pervasive caveat:** **single seed.** Large, multi-dataset-consistent effects (the
  ≥.03 ones, the LP synergy, the ablation collapses) are trustworthy; sub-.02 gaps are
  not. A seed sweep is the top outstanding hardening for the whole program.

---

## 4. Results by thrust — raw data

### A. Graph diagnostics — what differs, and does it predict transfer?

`graph_divergence` separates three axes (topology / feature marginals /
feature–structure coupling) and finds retweet graphs are **not feature-smooth over
edges** (edge-vs-random feature-cosine gap only 0.015–0.070 on every graph).
`similarity_vs_transfer` joins those divergences to a 4-source NM transfer matrix
(within-target Spearman, sign expected negative):

| Similarity axis | mean ρ (self-excl.) | sign-consistent |
|---|---:|---:|
| **proxy_a_distance** (feature-cloud separability) | **−0.86** | 5/5 |
| homophily_gap (signed coupling) | −0.80 | 5/5 |
| feat_frechet / feat_mmd2 | −0.66 | 4/5 |
| indegree_ks / outdegree_ks (**topology**) | −0.46 / −0.42 | 4/5 |

**More-divergent source ⇒ worse transfer, and it's the *feature* axis doing the
predicting.** `cp_hk` is the lone topology sign-flip anomaly (high-reciprocity outlier).
*Caveat: N=20 directed pairs, descriptive not powered; a training-family confound
motivates the interventional sweep.*

The node-level follow-up shows that sampled neighborhood context amplifies this
domain signal before learning: across all 28 graph pairs, a held-out linear domain
classifier rises from 0.794 accuracy / proxy-A 1.175 on raw centers to 0.899 / 1.597
on neighbor means and 0.907 / 1.629 on center+neighbor information. The latter is an
information-level diagnostic of the two channels available to SAGE, not a literal
1,536-dimensional model input. Label separability follows label homophily rather
than source separability uniformly: node+neighbor ROC-AUC is 0.982 for Election
ideology, 0.892 for COVID Political ideology, 0.700 for TwiBot bot status, and 0.554
for Ukraine suspension. Ukraine Suspended's label assortativity is effectively zero
(0.004), explaining why neighborhood averaging provides no label benefit even though
it makes graph source easier to identify. See
[`path_feature_coupling`](../../graphs/structure_features/path_feature_coupling/FINDINGS.md).

### B. NM merged-vs-single — the cross-source shortcut

The motivating "inversion" (single-source beats merged cross-domain) **does not
reproduce** once architecture / budget / eval are held fixed. ukr/covid, accuracy
@match (matched total compute):

| train | test:ukr | test:covid |
|---|---:|---:|
| single ukr (in-domain) | 0.515 | 0.614 |
| single covid (in-domain) | 0.459 | 0.664 |
| merged proportional | 0.479 | 0.637 |
| **merged within-source** | **0.500** | **0.659** |

- **No inversion:** merged ≥ single cross-domain at matched compute; the original result
  was an unfair arch/aug mismatch + a degenerate 0-shot eval.
- **Within-source > proportional** at both compute levels (+.02 @match).
- **Size imbalance (covid/midterm):** naive merged *collapses* on the tiny domain
  (midterm ~1.5% of merge: 0.31 vs single 0.417); **balanced within-source rescues it**
  to 0.405 @match / **0.427 @full — above the single-midterm specialist (0.417)**.
- **No OOD bonus** in either experiment: held-out transfer is carried by whichever big
  source is present; merging adds nothing out-of-distribution.

### C. NM transfer geometry — 8×8 matrix + interpolation ladder

**Single-source 8×8 (ROC-AUC, diagonal = in-domain specialist):** every graph has a
strong specialist (.906–.981) and **the specialist beats the all-8 merged model on all
8 columns** — Δ smallest on big twitter graphs (covid +.006, ukr +.013), largest on the
small/topical ones (**cp_hk +.039, ukr_susp +.033, elec20 +.032**). Donor ranking (mean
off-diagonal out): `ukr .849 ≈ covid .847 > twibot20 .817 > … > elec20 .649`; **covid is
the best cross-source donor to 7/8 targets**, and ukr↔covid transfer is nearly free
(ukr→covid .973 ≈ covid's .981 ceiling). **cp_hk is an island** — worst donor *and*
hardest target.

**Merged interpolation ladder (add one source per rung, test AUC):** a clean staircase —
each graph's column stays flat at its zero-shot level until it enters training, then
jumps and holds:

| rung enters | before → after | Δ |
|---|---|---:|
| covid_political | .830 → .911 | +.081 |
| election2020 | .830 → .926 | +.096 |
| **ukr_rus_suspended** | .769 → .934 | **+.165** |
| twibot20 | .924 → .938 | +.013 |
| **cp_hk** | .727 → .867 | **+.140** |

**Column, not count, drives AUC:** rung-7 ≈ rung-8 (all-8) on every column *except*
cp_hk. twibot20 barely moves (+.013) — a retweet graph NM already transfers to.
**The clean trade:** merging gives up ~.006–.04 of in-domain peak to buy +.09–.16 of
robustness on graphs with no strong donor.

### D. Features vs topology — what does NM use?

Eval-time input ablation on a fixed NM checkpoint (NM accuracy, chance ≈ 0.033):

| dataset | intact | zero | permute | **noise** |
|---|---|---|---|---|
| covid19 (in-domain) | 0.664 | 0.073 | 0.626 | **0.061** |
| midterm | 0.313 | 0.086 | 0.311 | **0.064** |
| twibot20 | 0.406 | 0.066 | 0.407 | **0.058** |

**`noise` (distinct but wrong content) collapses NM to chance, matching `zero`; only
`permute` (real content, scrambled binding) survives** ≈ intact. So NM uses real
neighborhood feature **content as a permutation-invariant bag**, not node-distinctness
and not topology. Features are genuinely informative (raw feature→label AUC 0.71–0.95),
and **downstream is content-driven too**: `noise` collapses the frozen-rep label probe
to chance on political graphs (covid_political 0.912→0.535) *with topology fully intact*.
*Caveat: n_hop=1 stars — whether real multi-hop structure adds signal needs retraining,
not an eval ablation.*

### E. SSL-objective studies — can one pretext do both?

**E-rotation (multitask_ssl):** rotate SSL *pretext tasks*
one-per-episode at matched 40k. Static-LP is the headline (0-shot AUC, chance 0.50):

| arm | k | cls AUC | reg ρ | **static-LP** | min(cls, sLP) |
|---|---|---:|---:|---:|---:|
| NM | 1 | **0.810** | −0.001 | 0.467 | 0.467 |
| CL | 1 | 0.638 | −0.128 | 0.332 | 0.332 |
| FP | 1 | 0.492 | **0.166** | 0.449 | 0.449 |
| NMCL | 2 | 0.800 | −0.144 | 0.305 | 0.305 |
| NMFP | 2 | 0.802 | −0.098 | 0.424 | 0.424 |
| CLFP | 2 | 0.601 | 0.110 | 0.227 | 0.227 |
| **MIX** | 3 | 0.795 | 0.097 | **0.759** | **0.759** |

**MIX is the only generalist** — only arm above chance on the topological axis (+0.293
over the best control on the joint bar), winning all 4 LP datasets (+0.269 mean, incl.
held-out twibot20), while the controls emit *degenerate constant predictors*. And it is
a **3-way synergy**: no pair clears chance, capability is non-monotonic (singles 0.42 →
pairs 0.32 → triple 0.76). Classification is an **NM** property preserved under
combination; regression a weak **FP** property.

**E-engineering (topology_feature_ssl):** attack topology via architecture / augmentation
/ objective on the NM base. The capabilities **split by arm** and the multi-head
objective **fails** (test, matched-40k):

| arm | lever | reg ρ | cls AUC | static-LP | min(cls, sLP) |
|---|---|---:|---:|---:|---:|
| B0 | control | −0.00 | 0.793 | 0.675 | 0.675 |
| B1 | aug (feat-shuffle) | −0.12 | 0.799 | 0.341 | 0.341 |
| **E1** | directed degree inputs | **0.14** | 0.778 | 0.657 | 0.657 |
| **E2** | count-aware PNA agg | −0.08 | 0.781 | **0.761** | **0.761** |
| E2b | E2 drop-BN | −0.00 | 0.784 | 0.401 | 0.401 |
| E4 | multi-head objective | −0.13 | 0.445 | 0.662 | 0.445 |
| E4r | multi-head, rotated | −0.12 | 0.643 | 0.234 | 0.234 |

**Feature-strong (E1: regression) and topology-strong (E2: static-LP) are *different*
arms; no engineered arm clears the joint bar.** The augmentation lever backfired (B1
LP below chance); "representable ≠ usable" (E2b makes counts linearly readable yet
crashes LP); and **E4 — the multi-head objective meant to unify them — degrades both**
(the heavy-tailed structural term dominated the loss; MFR's near-zero gradient couldn't
hold feature content).

> **The reconciliation (§5.5):** rotating heterogeneous *pretext tasks* (MIX) gets the
> generalist; hand-building a multi-head *objective* (E4) does not. Same goal, opposite
> outcome — the composition mechanism matters.

### F. Frozen-probe strategy benchmark

Anchored to floors (untrained `random_init`; `features_only` ridge). The prior
fine-tuning matrix (covid_task_transfer_matrix) was uninterpretable — full adaptation
drove every cell, including from-scratch, to ceiling — motivating the **frozen** probe.

**Static-LP (0-shot AUC):** `NM·covid` **0.612 (+0.229 over floor)**, above floor on all
5 datasets; every other encoder (CL, FP, **merged-NM 0.352**) sits *below the untrained
floor*. **Node regression (Spearman ρ):** `features_only` **0.109** ≫ `random_init`
0.022 ≫ every trained encoder (all ≤0, NM·covid −0.053). **Pretraining is strictly
harmful for regression; the GNN discards feature signal a plain ridge keeps.**

Takeaways: (1) frozen SSL transfers **only when the pretext structurally matches the
task** (adjacency prediction → link prediction); (2) **no objective is best across
tasks** (NM wins LP, is 2nd-worst on regression); (3) **merging destroyed the one thing
that worked** (merged-NM LP 0.352 vs single-covid 0.612) — consistent with the
within-source > merged pattern, though note this cuts against thrust B's transfer story
(see §5.6).

### G & H. Enablers — new tasks and datasets

- **node_regression** (6 exogenous profile targets, log1p) and **static_link_prediction**
  (present-vs-absent edges, no temporal split) were built to give the frozen-probe evals
  **real headroom** — they are the feature-secondary and topological axes used throughout
  thrusts E/F.
- **twibot20** (reconstructed retweet graph, 162,990 nodes, bot-vs-human) enters as a
  transfer domain: it's an **easy target** (NM pretrained on twitter retweet graphs hits
  ~.92 zero-shot) and a solid donor (mean out .817). **cp_hk** (COSINE 2022 HK/China
  political) enters as the **isolated island** — the consistent worst donor and hardest
  target across every matrix. Their notable findings live in thrusts A/C, not as
  standalone results.

---

## 5. Findings / discussion

Cross-cutting, evidence-based headlines (each tied to ≥1 thrust above; all 1 seed):

1. **NM ≙ feature-content matching.** Ablation (D) and similarity-vs-transfer (A)
   independently show the signal is feature content, not topology — and transfer is
   predicted by feature-cloud separability, not degree distributions. *This is the single
   most robust finding in the program (two orthogonal methods).*

2. **Merged ≥ single on transfer, but merging is a robustness trade, not a free lunch.**
   The inversion is dead (B); merging costs ~.006–.04 in-domain (paid by small graphs)
   to buy +.09–.16 where no single donor exists (C). It **never** helps out-of-domain.

3. **Within-source, balanced sampling is the right default for merged NM.** Removing the
   cross-source shortcut helps consistently, and balancing rescues starved small domains
   above their own specialist ceiling (B). The cross-source-probability sweep bottoms out
   at p=0.

4. **A both-tasks encoder is achievable only by pretext rotation, and only as a 3-way
   synergy.** MIX clears the joint bar (0.76); no single or pair does; capability is
   non-monotonic (E). This is the program's one clean *positive* generalist result.

5. **Composition beats construction.** Rotating existing pretexts (MIX) unlocks
   topological transfer; a purpose-built multi-head objective (E4) degrades the encoder
   below the NM control (E). If topology must be *used*, put diverse *tasks* in the
   rotation — don't hand-weight a structural loss (its heavy-tailed target dominates and
   the feature anchor is too weak to survive co-training). The follow-up is a
   **stronger-than-MFR feature-preservation objective** and/or uncertainty weighting.

6. **NM's link-prediction transfer is setup-sensitive — do not compare absolute LP
   numbers across experiments.** Frozen NM→LP reads 0.612 (single covid, step 11k, F),
   0.467 (merged, global, 30k, E-rotation), and 0.675 (merged, within-source, 40k,
   E-engineering B0). Corpus, sampling regime, and checkpoint each move it by >0.1, and
   F's "merging destroys LP" vs B's "merging ≥ single" are **not** a contradiction —
   they differ in task (LP vs NM retrieval), corpus, and step. **Only within-experiment
   contrasts are valid.** Reconciling these under one matched protocol is an open thread.

7. **For node regression, raw features win — pretraining is harmful.** Every frozen
   encoder underperforms a plain ridge on raw bios (F); NM even anti-scales on regression
   with budget (E-engineering budget sweep). Regression is the axis where the current
   stack adds nothing.

### Where this lands

The program has mapped the **current NM stack thoroughly**: we know it learns feature
content, how sources combine, which graphs donate and which are islands, and that
neither augmentation nor a hand-built multi-task loss makes it topological. The **one
forward door that opened** is heterogeneous pretext rotation (MIX) — the only encoder
strong on both feature and topological tasks, emergent from the *complete* {NM,CL,FP}
set. The **two big caveats gating every claim**: (a) **single seed** everywhere — a seed
sweep is the top priority; (b) the **causal edge/feature ablation on MIX** (rewire edges
vs permute features) that would prove its LP win is *topological* rather than a feature
artifact is teed up but unrun. Secondary open threads: a matched-*per-task* (120k) MIX to
separate the mixing effect from the ⅓-compute dilution; the interventional single-axis
similarity sweep; and a unified protocol to reconcile the setup-sensitive NM→LP numbers
(§5.6).

---

_Per-thrust write-ups are the source of truth for every number here; this document only
consolidates their headlines. Structure mirrors
[`topology_feature_ssl/FINDINGS.md`](../../objectives/topology_vs_features/topology_feature_ssl/FINDINGS.md). Earlier
cross-experiment summaries:
[`NM_MERGED_VS_SINGLE_SUMMARY.md`](./NM_MERGED_VS_SINGLE_SUMMARY.md),
[`NM_CROSS_SOURCE_STUDY.md`](./NM_CROSS_SOURCE_STUDY.md)._
