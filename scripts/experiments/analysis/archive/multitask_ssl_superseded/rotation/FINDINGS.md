# Findings — Multi-task SSL rotation (one encoder, all SSL tasks)

> **⚠️ SUPERSEDED — do not cite.** Every static-link-prediction number below came
> from an evaluator found to be invalid on 2026-07-23 (center-blind scoring, frozen
> random prototypes, degree-confounded negatives). The rescore of the same frozen
> checkpoints **inverts the headline**: link prediction is a neighbor-matching main
> effect that rotation dilutes — NM is the best arm on all 5 datasets and MIX sits
> below the heuristic floors. There is no 3-way synergy.
>
> Current read: [`../../multitask_ssl/FINDINGS.md`](../../multitask_ssl/FINDINGS.md) ·
> defect details: [`../../multitask_ssl/FINDINGS_rescore.md`](../../multitask_ssl/FINDINGS_rescore.md)
>
> The classification and regression sections never touched the broken path and remain
> valid; they are consolidated into `multitask_ssl/`.


**Design:** 4 arms, 1 seed, matched total compute. Three single-objective controls
(NM, CL, FP) vs one **rotation treatment (MIX)** that cycles all three objectives
one-per-episode. Frozen-encoder transfer to node classification, node regression, and
static link prediction. Built ground-up from the raw eval CSVs (see [Results](#results)).

---

## 1. Executive summary

Rotating an encoder over **all three** SSL objectives — one objective per episode,
1:1:1, at the *same total compute* as each single-objective control — produces the
**only representation that is good on both feature tasks and topological tasks at
once**. No single objective does this: each pure control is a specialist that collapses
to chance on at least one task family.

The headline is **emergent topological transfer**. On static link prediction (0-shot,
frozen embeddings), MIX reaches **0.759 mean ROC-AUC** while every single-objective
control sits **at or below chance** (NM 0.467, CL 0.332, FP 0.449). MIX beats the best
control on **all four** LP datasets: mean **+0.269**, range **[+0.19, +0.33]**, all
positive — including the fully held-out `twibot20` (+0.254). The controls don't just
score lower; they emit **degenerate constant predictors** (accuracy ≈ 0.50, f1 = 0.0
or 0.667), whereas MIX genuinely separates edges (accuracy 0.64–0.78).

MIX buys this generalization **nearly for free** on the feature specialties: it ties
the best control on classification (0.795 vs NM 0.810; tied at ceiling on
`election2020`) and is a positive-signal second on regression (0.097 vs FP's 0.166,
while NM/CL are ≈0 or negative).

**Bottom line (1 seed):** at matched compute, rotation is a Pareto win on the *joint*
criterion — the only arm above chance on the topological axis while staying near-best
on the feature axis. Specialists still win their own single task, so this is "best
general encoder," not "free lunch on every task."

![Capability plane: feature (classification AUC) vs topological (static-LP AUC) transfer. Only MIX sits in the top-right generalist quadrant; NM/CL/FP are stranded at or below the topological-chance line.](figures/0_capability_plane.png)

*Headline figure — the capability plane. x = feature transfer (classification AUC),
y = topological transfer (static-LP AUC); chance = 0.50 on both. Only MIX (rotation)
reaches the top-right "good at both" quadrant; every single-objective control is
pinned at/below the topological-chance floor. Marker size ∝ regression ρ (the secondary
feature axis, where FP is the specialist); whiskers = static-LP min–max across the 4
eval datasets. Built by `plot_capability_plane.py` from the raw CSVs.*

---

## 2. Methodology

### Central hypothesis

An encoder pretrained by **rotating over a heterogeneous menu of SSL objectives**
(one objective per episode, shared weights) will yield a frozen representation that
transfers **better across all downstream task families** than an encoder trained on any
**single** SSL objective — because different objectives inject different inductive
biases (feature content vs. neighborhood/topology vs. generative reconstruction), and a
single encoder forced to satisfy all of them cannot overfit the shortcut of any one.

The comparison is scored on a **joint** criterion — `min(feature_score,
topological_score)` — never a single averaged number. An encoder that is excellent at
one family and useless at another is not a good general encoder, and a mean would hide
that.

### The three objectives and the motivation for each

All three are defined on the retweet graphs and share the same metric-episode structure
(30-way, 3-shot, 4-query), so they compose cleanly on one shared encoder. They run
through the *identical* forward pass — only the augmentation and the loss differ per
episode.

| arm | objective | what it should teach | why include it (hypothesis) |
|---|---|---|---|
| **NM** | `neighbor_matching` — instance/neighborhood discrimination, no augmentation, metric loss | which nodes are neighbors → local topology **and** a feature shortcut | Strong on feature/identity tasks; a natural topology signal. Prior work in this repo shows NM leans heavily on feature *content*. Expected to be a **classification specialist**. |
| **CL** | `contrastive` — two feature-augmented views (NZ0.2 = zero 20% of feature dims), metric loss | invariance to feature corruption (two-view instance discrimination) | Should sharpen feature representations robust to noise. Risk flagged up front: NZ0.2 may be **too easy** (pretext saturates), giving weak features. |
| **FP** | `masked_feature_prediction` — GraphMAE-style, mask 30% of node features (zero), reconstruct via an aux MSE head | reconstruct continuous node features → generative, content-preserving | Generative objectives preserve continuous feature variation that discrimination throws away → expected **regression specialist**, and the complement to NM/CL. |
| **MIX** | `nm_fp_cl` rotation, one objective per episode, round-robin 1:1:1 | union of all three biases on one shared encoder | **The treatment.** If rotation inherits each objective's strength, MIX should be the only arm strong on every task family. |

**Motivation for the treatment (MIX):** the three controls are hypothesized to be
*complementary specialists* (NM→discrimination, FP→reconstruction, CL→robust features).
The question is whether a single encoder, exposed to all three in rotation, composes
their strengths into one general representation — or whether mixing merely dilutes each.

### Rotation mechanism

MIX uses PRODIGY's `MultiTaskSplitBatch([NeighborTask, ContrastiveTask,
ContrastiveTask], ["nm","cl","fp"], counts)`. Each episode is assigned one task by a
count-weighted round-robin; the `Collator` reads the task tag and (a) applies the
per-task augmentation (nm→identity, cl→NZ0.2, fp→NZ0.3) and (b) sets `graph.mix_is_fp`
so the trainer dispatches the reconstruction loss on fp episodes and the metric loss on
nm/cl episodes. `batch_size: 1` makes the per-episode task/loss dispatch exact.
Validation falls back to a pure-NM monitor (mixing recon and metric scores in one
accumulated number is meaningless); the real comparison is the downstream sweep.

### Compute matching (important caveat baked into the design)

All arms train the **same total budget (40k episodes)**. MIX therefore sees only **~⅓
the per-task exposure** (~13.3k episodes each) of a single-objective control. This is
the intended "same compute, mixed vs pure" test, but it means **any MIX win is a lower
bound** — a matched-*per-task* (120k) MIX is the natural follow-up and is deferred.

---

## 3. Experiment setup

**Fixed across all four arms (single-variable = SSL objective only):**

- **Pretrain corpus:** the 3-way merged retweet graph
  `/dataMeR1/phil/data/merged/graphs/ukr_rus_covid_midterm_retweet_graph.pt`
  (`ukr_rus` + `covid19` + `midterm`), one seed.
- **Encoder:** bio-only 768-d GTE text embeddings → mean-aggregating GraphSAGE,
  1 layer / 1 hop, emb_dim 256, undirected message passing. **No structural inputs.**
  All four arms share this exact encoder.
- **Episode sampling:** **global** (no within-source confinement). `ContrastiveTask`
  (CL/FP) has no strata support, so global is the only regime all three objectives can
  share; NM uses it too to keep the arms single-variable. (⇒ `mtr_NM` here is *not* the
  same encoder as `topology_feature_ssl`'s within-source B0 — controls are internal.)
- **Budget:** 40k episodes (epochs 4 × dataset_len_cap 10k). Checkpoints at 10k/20k/30k.
- **Frozen checkpoint evaluated:** **`state_dict_30000.ckpt` for all four arms** — the
  highest checkpoint the 40k budget actually writes (the trainer saves no ckpt at the
  final 0-indexed step 39999). The comparison is thus matched at 30k across arms.

**Evaluation — frozen-encoder benchmark, one sweep per arm** (keyed by `model = arm`):

| task | protocol | metric | eval datasets |
|---|---|---|---|
| Node classification | 10-shot | ROC-AUC | `twibot20`, `election2020` — **both held-out** |
| Node regression | 10-shot, 3 targets (followers, statuses, account-age; log1p) | Spearman ρ | `midterm`, `ukr_rus`, `covid19` (in-domain) + `twibot20` (held-out) |
| Static link prediction | 0-shot | ROC-AUC | `midterm`, `ukr_rus`, `covid19` (in-domain) + `twibot20` (held-out) |

The three in-domain sources are *in* the pretrain mix, so treat them as fit; the clean
**transfer** read is the held-out datasets (`twibot20`, `election2020`). Note the
classification read is entirely held-out.

---

## 4. Results

> **Raw data (single source of truth):**
> `scripts/experiments/analysis/multitask_ssl_rotation/data/*.csv`
> (per-arm × per-dataset × per-target eval rows, test split; pulled from the run
> worktree).
> **Reproduce every number below:**
> `python scripts/experiments/analysis/multitask_ssl_rotation/aggregate_results.py --plotting-root scripts/experiments/analysis/multitask_ssl_rotation`
> **Figures:** `figures/0_capability_plane.png` (headline; built by
> `plot_capability_plane.py`) + per-task detail
> `figures/{1_regression,2_static_link_prediction,3_classification}.png`

### T1 — frozen-encoder transfer (mean over datasets, test split)

| arm | classification AUC | regression ρ | static-LP AUC | role |
|---|---|---|---|---|
| NM  | **0.810** | −0.001 | 0.467 | control (classification specialist) |
| CL  | 0.638 | −0.128 | 0.332 | control (weak everywhere) |
| FP  | 0.492 | **0.166** | 0.449 | control (regression specialist) |
| **MIX** | 0.795 | 0.097 | **0.759** | **rotation (treatment)** |

`MIX − max(NM,CL,FP)` per task: **classification −0.015** (vs NM), **regression −0.068**
(vs FP), **static-LP +0.293** (vs NM).

### Joint generalist bar — `min(feature = cls AUC, topological = sLP AUC)`

(chance = 0.50 on both axes; regression excluded as a secondary, noisy feature axis)

| arm | min(cls, sLP) | bottleneck |
|---|---|---|
| NM  | 0.467 | static-LP (at chance) |
| CL  | 0.332 | static-LP (at chance) |
| FP  | 0.449 | static-LP (at chance) |
| **MIX** | **0.759** | — (generalist) |

**MIX min − best-control min = +0.293.** Every pure objective bottoms out at chance on
the topological axis; only MIX clears it while staying near the top of the feature axis.

### Headline — static link prediction, ROC-AUC per dataset (0-shot)

| dataset | NM | CL | FP | **MIX** | MIX − max(control) |
|---|---|---|---|---|---|
| midterm (in-domain)         | 0.487 | 0.417 | 0.433 | **0.676** | +0.189 |
| ukr_rus (in-domain)         | 0.484 | 0.288 | 0.534 | **0.861** | +0.327 |
| covid19 (in-domain)         | 0.406 | 0.366 | 0.449 | **0.755** | +0.306 |
| twibot20 (**held-out**)     | 0.491 | 0.259 | 0.381 | **0.745** | +0.254 |
| **mean** | 0.467 | 0.332 | 0.449 | **0.759** | **+0.269** (range [+0.19, +0.33], all +) |

MIX static-LP AUC: mean **0.759 ± 0.066** (sd across datasets), range [0.676, 0.861].
**Mechanistic detail:** on static-LP the controls emit degenerate constant predictors
(accuracy ≈ 0.50; NM f1 = 0.0 = predict-all-negative, CL/FP f1 ≈ 0.667 =
predict-all-positive), i.e. their frozen embeddings carry **no linearly decodable
adjacency**. MIX's accuracy is **0.64–0.78** with real f1 — it actually separates
edges from non-edges.

### Classification, ROC-AUC per dataset (10-shot, both held-out)

| dataset | NM | CL | FP | **MIX** |
|---|---|---|---|---|
| twibot20      | 0.640 | 0.611 | 0.490 | 0.610 |
| election2020  | 0.980 | 0.665 | 0.495 | **0.981** |

MIX's 0.015 classification deficit vs NM comes entirely from `twibot20` (0.610 vs
0.640); on `election2020` MIX and NM are tied at ceiling (~0.98). FP is at chance on
classification (≈0.49).

### Regression, Spearman ρ per dataset (10-shot, mean over 3 targets)

| dataset | NM | CL | FP | **MIX** |
|---|---|---|---|---|
| midterm (in-domain)     | −0.044 | −0.156 | 0.181 | 0.178 |
| ukr_rus (in-domain)     | −0.024 | −0.131 | 0.201 | 0.132 |
| covid19 (in-domain)     | −0.020 | −0.158 | 0.177 | 0.119 |
| twibot20 (**held-out**) | 0.087  | −0.067 | 0.105 | −0.039 |

FP is the regression specialist everywhere; MIX tracks it closely in-domain (e.g.
midterm 0.178 vs 0.181) but **drops to slightly negative on held-out `twibot20`
(−0.039)** — the one place the ⅓-compute dilution appears to bite. NM ≈ 0 (no signal);
CL is negative throughout.

---

## 5. Findings / discussion

Evidence-based headlines (1 seed):

1. **Rotation is the only generalist.** On the joint `min(feature, topological)` bar,
   MIX scores 0.759 vs ≤0.467 for every single-objective control (**+0.293** over the
   best). Each pure objective collapses to chance on the topological axis; MIX is the
   only arm above chance on both axes at once.

2. **Emergent topological transfer is the result.** Static link prediction from frozen
   embeddings measures whether adjacency is linearly decodable. **Only MIX encodes it**
   (0.759 mean AUC; wins all 4 LP datasets, +0.269 mean, all positive, incl. held-out
   twibot20). This is not a small margin over a weak baseline — the controls are
   *degenerate constant predictors* (acc ≈ 0.50). Rotation composes a capability
   (decodable graph structure) that **none of NM, CL, or FP produces on its own**.

3. **Generalization is nearly free on the feature specialties.** MIX ties the best
   control on classification (−0.015; tied at ceiling on election2020) and is a
   positive-signal second on regression (0.097, vs NM/CL ≈ 0 or negative). Whatever
   rotation gives up on the pure feature tasks is marginal.

4. **Each pure objective is a collapsing specialist.** NM = classification-only
   (0.810 cls, ~0 reg, chance LP). FP = regression-only (0.166 reg, chance cls,
   chance LP). CL = weak everywhere (worst LP at 0.332) — consistent with the
   pre-registered worry that NZ0.2 contrastive is **too easy** and yields poor
   features. Picking any single objective forfeits at least one task family.

5. **It is a Pareto win, not a free lunch.** Specialists still beat MIX on their own
   task (NM cls +0.015, FP reg +0.068). The claim is narrow and defensible: rotation
   yields **the best *general* encoder**, dominating on the joint criterion, while a
   specialist remains better if you only care about its one task.

**Caveats / scope.** (a) **Single seed** — the LP win is large and consistent across
all 4 datasets (all positive, incl. held-out), so it is very likely real; the
regression deltas are within plausible noise. (b) **Matched *total* compute** — MIX had
~⅓ the per-task exposure of each control, so its wins are a **lower bound**; the one
soft spot (regression vs FP, and MIX going slightly negative on held-out twibot20) is
exactly what dilution would predict. A matched-per-task 120k MIX would separate the
compute effect from the mixing effect. (c) The static-LP result establishes that MIX's
embeddings *carry* decodable topology; the eval-time 2×2 ablation (rewire edges vs
permute features) that would pin the win causally on **adjacency** is teed up but **not
yet run**.
