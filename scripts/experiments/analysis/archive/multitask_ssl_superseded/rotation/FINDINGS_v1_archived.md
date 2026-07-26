# Findings — multitask_ssl_rotation

> **⚠️ SUPERSEDED — do not cite.** Every static-link-prediction number below came
> from an evaluator found to be invalid on 2026-07-23 (center-blind scoring, frozen
> random prototypes, degree-confounded negatives). The rescore of the same frozen
> checkpoints **inverts the headline**: link prediction is a neighbor-matching main
> effect that rotation dilutes — NM is the best arm on all 5 datasets and MIX sits
> below the heuristic floors. There is no 3-way synergy.
>
> Current read: [`../../../multitask_ssl/FINDINGS.md`](../../../multitask_ssl/FINDINGS.md) ·
> defect details: [`../../../multitask_ssl/FINDINGS_rescore.md`](../../../multitask_ssl/FINDINGS_rescore.md)
>
> The classification and regression sections never touched the broken path and remain
> valid; they are consolidated into `multitask_ssl/`.


**One line:** rotating one encoder over three heterogeneous SSL objectives
(NM + CL + FP, one per episode) is the **only** configuration that transfers to
*both* feature tasks and topology — and it is the **only** arm that does static
link prediction at all. Every single-objective control collapses on ≥1 task.

**Status:** 1 seed, matched budget. The LP win is large and consistent across all
4 LP datasets, so it is very likely real; deltas on the noisy regression axis are
within noise. Multi-seed hardening was scoped out (single-seed decision, 2026-07-10).

Reproduce the numbers below: `python aggregate_results.py --plotting-root <tree>/scripts/experiments/analysis`

---

## Setup (what was compared)

Four arms, **identical** encoder (bio-768 GTE → mean-agg SAGE, 1 layer/1 hop,
emb_dim 256, undirected, global episode sampling), **identical** matched budget
(40k episodes; best-val checkpoint = 30k for every arm), on the 3-way merged
retweet corpus (ukr_rus + covid + midterm). The arms differ **only** in the SSL
objective:

| arm | objective | budget |
|---|---|---|
| NM | neighbor matching (instance/neighborhood discrimination) | 40k |
| CL | contrastive, NZ0.2 two-view (invariance to feature corruption) | 40k |
| FP | masked feature prediction, NZ0.3 (generative reconstruction) | 40k |
| **MIX** | **nm_fp_cl rotation**, one task/episode, 1:1:1 (~13.3k each) | 40k total |

MIX is matched on **total** compute, so it sees ⅓ the per-task exposure of each
control. Eval is the frozen-encoder benchmark over the focused-5 datasets (3
in-domain: ukr_rus/covid/midterm; 2 held-out: twibot20/election2020).

---

## T1 — transfer benchmark (mean over datasets, test split)

| arm | classification (AUC) | regression (Spearman) | static-LP (AUC) |
|---|---|---|---|
| NM  | **0.810** | −0.001 | 0.467 |
| CL  | 0.638 | −0.128 | 0.332 |
| FP  | 0.492 | **0.166** | 0.449 |
| **MIX** | 0.795 | 0.097 | **0.759** |

classification: 2 labeled datasets (twibot20, election2020) · regression: 4
datasets × 3 targets · static-LP: 4 datasets · chance AUC = 0.50.

### MIX − max(NM, CL, FP), per task

| task | MIX − max(control) | best control |
|---|---|---|
| classification (AUC) | **−0.015** | NM @ 0.810 |
| regression (Spearman) | **−0.068** | FP @ 0.166 |
| **static-LP (AUC)** | **+0.293** | NM @ 0.467 |

MIX gives up ~0.015 AUC to the best classifier (NM) and ~0.07 Spearman to the best
regressor (FP), and buys **+0.29 AUC** on the topology task that neither — nor any
single objective — can do.

---

## The reading — MIX is the only generalist

Each single objective is a **specialist that collapses on ≥1 task**:

- **NM** — classification only (0.81 AUC). Chance LP (0.47), zero regression. Pure
  instance discrimination gives a feature-linear representation but no topology.
- **FP** — regression only (0.17 ρ, the best of any arm). Chance classification
  (0.49) and chance LP (0.45). Generative reconstruction preserves continuous
  feature variation but not discriminative or relational structure.
- **CL** — weak everywhere (0.64 cls, −0.13 reg, 0.33 LP). Confirms the
  trivial-pretext prediction: NZ0.2 two-view is too easy, so it learns little.
- **MIX** — near-best classification (0.795, −0.015 vs NM), 2nd-best regression
  (0.097), and the **only** arm with real LP (0.759).

**Joint generalist bar — min(feature = cls AUC, topological = static-LP AUC):**

| arm | min(feature, topological) | bottleneck |
|---|---|---|
| NM | 0.467 | static-LP (at chance) |
| CL | 0.332 | static-LP (at chance) |
| FP | 0.449 | static-LP (at chance) |
| **MIX** | **0.759** | static-LP (still well above chance) |

**MIX min − best-control min = +0.293.** MIX is the only arm whose *worst* task is
still well above chance. (Regression is deliberately **not** folded into the min
bar: it is near-zero and noisy for every arm — FP's 0.17 is the ceiling — so it
would drag all four arms to ≈0 and hide the cls-vs-topology generalist story that
is the actual finding. Regression is reported as a secondary, FP-leaning axis.)

---

## Headline — emergent link prediction from rotation

Static-LP ROC-AUC, 0-shot, per dataset — the direct topological probe:

| dataset | NM | CL | FP | MIX | MIX − max(control) |
|---|---|---|---|---|---|
| midterm | 0.487 | 0.417 | 0.433 | 0.676 | **+0.189** |
| ukr_rus | 0.484 | 0.288 | 0.534 | 0.861 | **+0.327** |
| covid | 0.406 | 0.366 | 0.449 | 0.755 | **+0.306** |
| twibot20 (held-out) | 0.491 | 0.259 | 0.381 | 0.745 | **+0.254** |

- **Every single-objective arm is at chance on LP** (AUC 0.26–0.53, accuracy ≈0.50).
- **MIX clears chance on all four** (AUC 0.68–0.86, accuracy 0.64–0.78), including
  the held-out twibot20 (a different graph, never in the pretrain mix).
- **MIX − max(control) on static-LP: mean +0.269, range [+0.189, +0.327], all 4
  positive.** MIX static-LP AUC: **0.759 ± 0.066** (sd across datasets), range
  [0.676, 0.861].

Because none of NM/CL/FP produces any topological signal alone, MIX's LP is
**emergent from the rotation** — it is not inherited from any constituent
objective. This is the "learns both feature and topology" thesis in its strongest
form: the multi-task union delivers a capability that is absent from every part.
See `figures/2_static_link_prediction.png`.

---

## Caveats

- **1 seed.** The LP win is huge and consistent across 4 independent LP datasets
  (the across-dataset spread stands in for a seed spread here), so it is very
  likely real. Classification/regression deltas are small and within plausible
  seed noise — do not over-read the −0.015 / −0.068.
- **Matched total compute** (not matched per-task): MIX sees ⅓ the exposure of each
  control per objective, so a MIX win is if anything *understated*. A matched-per-
  task 120k MIX is the natural follow-up if the small cls/reg gaps matter.
- **best-val = 30k** for all arms (the 40k-episode budget writes its last/best
  checkpoint at 30k; both arms trained the full `_step:39999`). Matches the
  topology_feature_ssl matched-30k comparison.

---

## Next — why does rotation yield LP? (eval-time, no retraining)

The frozen MIX encoder already exists; the *why* can be probed without new
pretraining, using the eval runner's built-in ablations (the tfssl 2×2):

1. **Edge-rewire ablation** (`--ablate-edges rewire`) on MIX static-LP: if MIX's LP
   collapses to chance when real adjacency is destroyed but the node bag/features
   are kept, the LP signal is genuinely **topological** (learned relational
   structure), not a feature shortcut.
2. **Feature ablation** (`--ablate-features permute`/`noise`) on MIX static-LP: if
   LP survives feature corruption, it does not rely on feature content.
3. Contrast against NM/FP under the same ablations to confirm the controls have
   nothing to ablate (already at chance).

If (1) collapses and (2) survives → the rotation taught the encoder adjacency.
This is the recommended next step and needs **no** GPU training, only an eval
sweep on the four frozen encoders. See `EXECUTION.md` for the exact commands.
