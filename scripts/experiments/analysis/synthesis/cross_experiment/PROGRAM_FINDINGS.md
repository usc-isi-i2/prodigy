# GFM Retweet-Graph Program — Consolidated Findings

For the current hierarchical program overview, including later experiments, see
[Experiment overview: June–September 2026](../../EXPERIMENT_OVERVIEW.md).

**Can one pretrained encoder transfer across retweet graphs and downstream tasks?**
The early studies show strong task dependence. NM retrieval relies on neighborhood
feature content; source composition and episode sampling affect transfer. On the
corrected static-link evaluator, **NM leads the objective lattice on classification
and link prediction; rotating NM/CL/FP (MIX) trades some of that performance for
positive mean regression. There is no emergent three-way link-prediction synergy.**

_Status: historical synthesis of the early program, originally consolidated
2026-07-20; static-link correction completed 2026-09-11. This is not a synthesis of
all experiments completed since July; see the [current analysis index](../../README.md).
The studies summarized here are single-seed. Compare absolute values only within
matched experiments; corpora, sampling, checkpoints and evaluation protocols differ._

> **Correction record.** The old episodic static-link evaluator was endpoint-blind,
> used frozen random class prototypes, and had degree-confounded negatives. Its
> numbers and derived joint scores have been removed here. Section 4E uses the
> valid rescore of the same frozen checkpoints; sections 4E-engineering and 4F
> retain only unaffected classification/regression results because the cited
> rescore does not cover those arms. Temporal LP has the same defect and remains
> unrescored in this evidence set. See the
> [corrected lattice](../../objectives/multitask/multitask_ssl/FINDINGS.md) and
> [rescore protocol and results](../../objectives/multitask/multitask_ssl/FINDINGS_rescore.md).

---

## 1. Executive summary

The early studies support the following scoped findings:

- **NM retrieval depends on neighborhood feature content in the tested one-hop setup.** Destroying
  real neighborhood content collapses NM to chance exactly like deleting features, while
  scrambling the feature↔node binding is harmless (feature_ablation). Independently,
  **feature-cloud separability predicts transfer** (proxy-A-distance ρ≈−0.92) while raw
  degree-distribution distance is the weakest predictor (ρ≈−0.6) (similarity_vs_transfer).
  These diagnostics do not establish that NM cannot encode adjacency: valid pairwise
  link prediction shows that it can (§4E).

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

- **NM leads the corrected classification/link-prediction comparison.** In the
  seven-arm objective lattice, NM has mean classification AUC 0.810 and pairwise
  LP AUC 0.757; MIX has 0.795 and 0.680. MIX retains positive mean regression
  (0.097 versus NM −0.001), a possible compromise that needs replication.

- **Engineered objectives did not rescue classification/regression in these tests.**
  E4/E4r underperform the NM control on both retained metrics. In the separate
  frozen-probe benchmark, raw-feature ridge beats every tested encoder on regression.
  Neither study's old static-LP results can support an objective ranking.

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
| E | SSL-objective studies | [topology_feature_ssl](../../objectives/topology_vs_features/topology_feature_ssl/FINDINGS.md), [multitask_ssl](../../objectives/multitask/multitask_ssl/FINDINGS.md) | NM leads the corrected cls/LP lattice; MIX retains positive mean regression. E4 degrades cls/reg; its LP ranking is unverified. |
| F | Frozen-probe strategy benchmark | [pretrain_probe_matrix](../../objectives/frozen_probes/pretrain_probe_matrix/FINDINGS.md), [pretrain_strategy_benchmark](../../../setup/pretrain_strategy_benchmark/README.md), [covid_task_transfer_matrix](../../../setup/covid_task_transfer_matrix/README.md) | Raw features beat all tested encoders on regression; old LP comparisons are excluded. |
| G | New downstream tasks (enablers) | [node_regression](../../../setup/node_regression/README.md), [static_link_prediction](../../../setup/static_link_prediction/README.md) | Continuous + edge-level tasks with real headroom; used as the topology/feature axes above. |
| H | New datasets (enablers) | [twibot20_transfer](../../../setup/twibot20_transfer/README.md), [cp_hk_twitter](../../../setup/cp_hk_twitter/README.md), [cp_hk_transfer_in](../../../setup/cp_hk_transfer_in/README.md) | twibot20 = easy transfer target (~.92 zero-shot); cp_hk = the isolated island. |

### How to read this document

- **Report task scores separately.** A minimum of classification and valid LP AUC
  can summarize their tradeoff, but their dataset sets differ and it excludes regression.
  Feature axis = node-classification ROC-AUC (and node-regression Spearman ρ as a
  secondary, noisy axis); adjacency axis = valid pairwise static-LP ROC-AUC. Link prediction alone does
  not identify whether the signal comes from topology or features.
- **Floors matter.** "Improves" means *beats a floor* (`raw_feat` ridge with no GNN;
  `random_init` untrained encoder; `raw_degree` leakage control), not *closest to a
  saturated ceiling*.
- **Single seed:** small gaps have no established significance. Historical budget
  labels may exceed the actual checkpoint step; compare recorded checkpoints.

---

## 3. Experiment setup (shared)

- **Encoder (default):** 768-d GTE bio embeddings → mean-aggregating GraphSAGE,
  1 layer / 1 hop, emb_dim 256, undirected message passing. Structural-input and
  aggregation variants live in thrust E.
- **NM eval regime:** 30-way / 3-shot (always ≥3 shots — 0-shot NM has no support
  prototypes and collapses to chance, an eval artifact that once made every model look
  random). ROC-AUC is near-ceiling (~.9+) on retweet graphs, so **accuracy is the
  discriminative DV** for NM transfer; AUC is reported for LP/classification.
- **Retained node tasks:** classification (ROC-AUC, 10-shot) and regression
  (Spearman ρ, 10-shot, log1p on the profile panel).
- **Corrected static LP:** symmetric endpoint-embedding cosine on a background
  graph with holdout edges removed; validation-locked score orientation; 2,000
  positive and 2,000 degree-matched negative pairs per dataset, shared across arms.
  Heuristic and raw-feature floors use the same pairs. Five datasets are scored.
- **Checkpoint hygiene:** before the 2026-07-26 terminal-save fix, nominal 40k
  rotation runs ended with a 30k checkpoint. The rescore reuses those checkpoints;
  it is not new training. Pin actual steps rather than inferring them from budget
  labels or selecting the highest checkpoint across pre-/post-fix runs.
- **Uncertainty:** these results have one training seed. Agreement across datasets
  supports the direction of a comparison but does not supply a seed confidence
  interval. Evaluation episodes are split-seeded; changing `--seed` does not
  resample them.

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
in this one-hop retrieval test; this does not rule out adjacency information
elsewhere in the representation. Features are informative (raw feature→label AUC 0.71–0.95),
and **downstream is content-driven too**: `noise` collapses the frozen-rep label probe
to chance on political graphs (covid_political 0.912→0.535) *with topology fully intact*.
*Caveat: n_hop=1 stars — whether real multi-hop structure adds signal needs retraining,
not an eval ablation.*

### E. SSL-objective studies — can one pretext do both?

**E-rotation (multitask_ssl):** seven objective arms, with the original frozen
checkpoints rescored on valid pairwise static LP. Classification averages two datasets,
regression four, and LP five; these means are not measurements on one shared target set.

| arm | k | cls AUC | reg ρ | valid LP AUC | LP margin over floor |
|---|---|---:|---:|---:|---:|
| NM | 1 | **0.810** | −0.001 | **0.757** | **+0.113** |
| CL | 1 | 0.638 | −0.128 | 0.543 | −0.101 |
| FP | 1 | 0.492 | **0.166** | 0.499 | −0.145 |
| NMCL | 2 | 0.800 | −0.144 | 0.679 | +0.035 |
| NMFP | 2 | 0.802 | −0.098 | 0.738 | +0.094 |
| CLFP | 2 | 0.601 | 0.110 | 0.543 | −0.101 |
| MIX | 3 | 0.795 | 0.097 | 0.680 | +0.036 |

Source: [valid lattice and data provenance](../../objectives/multitask/multitask_ssl/FINDINGS.md).
The floor margin is the source's mean margin against the best per-dataset heuristic;
it is not a comparison with the old random-prototype decoder.

**NM beats MIX on LP in all five datasets.** NM-containing arms clear the mean
heuristic floor; arms without NM do not. Adding objectives to NM lowers mean LP
in these runs, with NMFP retaining more than NMCL or MIX. This does not isolate
compute dilution from objective interference. MIX's positive regression is a
possible breadth tradeoff, not evidence of emergent LP capability; its regression
sign needs replication (one seed, four noisy datasets).

The corpus replications also rank NM > MIX > CL > FP by mean valid LP:
all-eight NM/MIX = 0.744/0.692; COVID-only NM/MIX = 0.728/0.652.
These corroborate the objective ordering, but do not isolate corpus effects.
The three-source NM arm transfers to held-out TwiBot20 at 0.835 versus a best
heuristic floor of 0.726. HK remains weak across arms. This is adjacency-prediction
evidence; a causal topology-versus-feature attribution requires separate controls.
See the [15-arm rescore](../../objectives/multitask/multitask_ssl/FINDINGS_rescore.md).

**E-engineering (topology_feature_ssl):** only unaffected node-task values are
retained from the historical comparison:

| arm | lever | reg ρ | cls AUC |
|---|---|---:|---:|
| B0 | control | −0.00 | 0.793 |
| B1 | aug (feat-shuffle) | −0.12 | 0.799 |
| E1 | directed degree inputs | **0.14** | 0.778 |
| E2 | count-aware PNA agg | −0.08 | 0.781 |
| E2b | E2 drop-BN | −0.00 | 0.784 |
| E4 | multi-head objective | −0.13 | 0.445 |
| E4r | multi-head, rotated | −0.12 | 0.643 |

E1 has the strongest mean regression here; E4/E4r reduce classification and
regression relative to B0. The old LP columns cannot establish that E2 is the
best topology arm, that B1 destroys LP, or that no engineered arm clears a joint
classification/LP bar. Those comparisons require a valid rescore of these specific
checkpoints. The [source write-up](../../objectives/topology_vs_features/topology_feature_ssl/FINDINGS.md)
still contains invalid LP claims and is used here only for node-task results.

### F. Frozen-probe strategy benchmark

The retained regression comparison is `features_only` ridge **0.109**, untrained
`random_init` **0.022**, and all tested trained encoders ≤0 (NM·COVID −0.053).
Thus pretraining hurts regression **in this benchmark**, relative to raw features.
This is not a universal claim about all arms: the separate objective lattice has
positive regression for FP, CLFP and MIX.

The old LP ranking, the claimed single-source NM advantage over the random encoder,
and the claim that merging destroys LP are excluded. These checkpoints are not
covered by the cited 15-arm rescore, so corrected scores from other runs cannot
replace them. The [original benchmark](../../objectives/frozen_probes/pretrain_probe_matrix/FINDINGS.md)
remains a source for regression only until its LP comparison is rescored.

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

1. **NM retrieval uses neighborhood feature content in the tested setup.**
   Ablation (D) and similarity-vs-transfer (A) support this restricted conclusion.
   The corrected LP result shows useful adjacency information; it does not by itself
   distinguish feature-driven from topology-driven prediction.

2. **Merged ≥ single on transfer, but merging is a robustness trade, not a free lunch.**
   The inversion is dead (B); merging costs ~.006–.04 in-domain (paid by small graphs)
   to buy +.09–.16 where no single donor exists (C). It **never** helps out-of-domain.

3. **Within-source, balanced sampling is the right default for merged NM.** Removing the
   cross-source shortcut helps consistently, and balancing rescues starved small domains
   above their own specialist ceiling (B). The cross-source-probability sweep bottoms out
   at p=0.

4. **The objective lattice favors NM on classification and LP.** NM scores
   0.810/0.757 versus MIX 0.795/0.680. The prior three-way-synergy interpretation
   was an evaluator artifact. MIX's positive mean regression is a tentative
   compromise, not an LP breakthrough.

5. **The engineering comparison supports a node-task conclusion only.** E4/E4r
   reduce cls/reg relative to B0. Their invalid LP measurements cannot establish
   a rotation-versus-joint-loss advantage or justify a topology recommendation.

6. **Compare valid, matched evaluations.** The rescore supports NM > MIX by mean
   LP across the three tested corpora. Historical LP values from other evaluators
   cannot quantify corpus, sampling or checkpoint effects. Re-evaluate the exact
   engineering/probe checkpoints under one protocol before making those claims.

7. **Raw features win regression in the frozen-probe benchmark.** This finding is
   local to that benchmark; positive regression in the separate lattice prevents
   a program-wide claim that every pretrained encoder is harmful.

### Remaining work for this historical evidence set

- Rescore the engineering and frozen-probe checkpoints before restoring their LP
  rankings or joint-task scores; revise their own source write-ups accordingly.
- Treat temporal-LP results as invalid until a valid endpoint-aware rescore exists.
- Replicate the objective lattice across training seeds, especially its regression
  signs, and test matched per-objective exposure to distinguish dilution from
  interference.
- Use edge/feature interventions on **NM's valid adjacency signal** to establish
  what drives it; the old proposed test of a MIX-only LP capability has no target.

These are limits of the evidence summarized here, not an inventory of all later
work. The [analysis index](../../README.md) tracks subsequent experiments.

---

_Per-thrust write-ups are the source of truth for every number here; this document only
consolidates their headlines. Structure mirrors
[`topology_feature_ssl/FINDINGS.md`](../../objectives/topology_vs_features/topology_feature_ssl/FINDINGS.md). Earlier
cross-experiment summaries:
[`NM_MERGED_VS_SINGLE_SUMMARY.md`](./NM_MERGED_VS_SINGLE_SUMMARY.md),
[`NM_CROSS_SOURCE_STUDY.md`](./NM_CROSS_SOURCE_STUDY.md)._
