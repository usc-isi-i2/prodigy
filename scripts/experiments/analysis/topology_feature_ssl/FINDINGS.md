# topology_feature_ssl — Findings

**One frozen SSL pretext that transfers to _both_ feature tasks and a topological
task?** No — across all three levers tried. At a matched 40k-episode budget: on the
**encoder axis** the feature and topological capabilities are carried by **different,
non-overlapping arms** (E1 vs E2); the **augmentation** lever (B1) backfired; and the
**objective** lever — the multi-task **E4** arm meant to unify them — **degrades both
tasks**. No single encoder clears the joint bar.

_Status: all three axes now run — B0, B1, E1, E2, E2b (encoder/aug) + **E4, E4r**
(objective, run 2026-07-13), single seed. All numbers below are re-derived ground-up
from the raw CSVs in `scripts/plotting/**/data/` (paths under each table)._

---

## 1. Executive summary

The experiment asks whether a self-supervised pretext can produce frozen
representations that are strong for **feature tasks** (node classification, node
regression) **and** a **topological task** (static link prediction) at the same time —
by learning topology *and* features, not features only (which is all Neighborhood
Matching / NM does today).

**Headline: the capabilities split by arm on the encoder axis, and the objective axis —
the multi-task arm designed to unify them — degrades both. The joint goal is unreached.**

- **E1** (inject directed in/out/log-degree as node inputs) is the **only** arm that
  makes NM's frozen reps useful for **regression** (only arm with positive Spearman on
  all six targets) — but it is middling on link prediction (0.657, below even the
  do-nothing control).
- **E2** (count-aware sum/PNA aggregation) is the **best** arm on **static link
  prediction** (0.761, clearly above all others) and leads the multi-neighbor
  capability probes — but its regression is *negative*, worse than no pretraining.
- **E4 / E4r** (multi-task MFR ⊕ directed-LP ⊕ structural, the objective lever) **fail
  both**: regression negative (E4 −0.13, E4r −0.12), classification crashed below the
  feature floor (E4 0.45, twibot20 0.38 < chance), static-LP no better than clean NM
  (E4 0.66 ≤ E2 and B0; E4r 0.23, below chance). Their joint `min(cls, slp)` — 0.45 and
  0.23 — is **worse than every prior arm**.
- **No arm scores well on both, on any axis.** `min(feature, topological)` is low for
  every arm. The central hypothesis — one pretext strong on both — is **not supported**.

Supporting evidence-based takeaways (details in §5):

1. **Feature-shuffle augmentation (B1) backfired on everything** — regression went
   negative and link prediction collapsed *below chance* (0.341). The cheap
   "corrupt the shortcut" lever did not recover topology; it destroyed usable structure.
2. **Count-aware aggregation is what buys the topological task**, not structural
   inputs: E2's static-LP 0.761 ≫ E1 0.657 ≈ B0 0.675, and only E2 lifts the
   existence/conjunction probes that require summation over neighbors.
3. **E1's regression win is at least partly real, not degree leakage**: its clearest
   gain is on `account_age_days` (0.118 vs feature-floor 0.024), a *content-linked*
   target that degree passthrough cannot explain.
4. **E2b confirms the BatchNorm mechanism but is not a fix**: dropping conv BatchNorm
   raises the count/degree linear-probes (out-degree 0.52→0.71) yet crashes static-LP
   to 0.401 — the count magnitude is *representable* in the frozen rep but *not usable*
   by the tasks.
5. **The objective axis (E4) was run — and it degrades the encoder rather than unifying
   it.** The multi-task arm underperforms even the NM control on both feature tasks and
   E2 on topology. Mechanism: even after clipping the heavy-tailed structural target, the
   structural-reconstruction term dominated the loss (weighted struct 1.5 ≫ lp 0.13 ≫
   mfr 0.02), so the encoder over-fit degree while the near-zero MFR gradient (GTE
   features are trivially reconstructed) couldn't hold feature content. Consistent with
   the free-preview (masked-feature ⊀ NM on regression, −0.016).

---

## 2. Methodology

### The question and the three levers

NM (the current pretext) is **feature-only by construction**: its positives are
same-neighborhood nodes that correlate through shared ego-graph *content* (chiefly the
anchor watermark). There is no topological solution to NM, and two structural facts
compound this: the background GNN hardcodes mean aggregation + mean readout (degree-
and count-blind), and the sampler symmetrizes edges (direction conflated). So topology
is neither *forced*, *representable*, nor *used*. Three levers attack these:

| Lever | Meaning | Arms |
|---|---|---|
| **Corrupt the feature shortcut** | force NM off content, cheapest | B1 |
| **Make topology representable** | structural inputs / count-aware aggregation | E1, E2 |
| **Make topology used** | change the objective (generative / multi-task) | E4, E4r |

### Treatments — hypothesis & motivation

- **B0 — control.** Mean-agg SAGE, bio features only, undirected message passing, no
  augmentation, NM objective. The reference every delta is read against.

- **B1 — feature-shortcut corruption** (augmentation lever). B0 + `NR0.3`: 30 % of
  nodes get a random *real* feature vector. *Motivation:* if NM leans on feature
  content, corrupting it should force topology use **without** the expensive objective
  change — the cheap test of whether the objective even needs to change. `NR` (random),
  not `NZ` (zeroing), because zeroing aliases the already-zero missing bios and gets
  ignored. *Hypothesis:* NM stops being chance under the random-feature ablation;
  regression rises. **If flat → the shortcut is not removable by augmentation, i.e. the
  objective must change.**

- **E1 — directed structural inputs** (encoder axis, objective still NM). Add per-node
  in-degree, out-degree, log-degree (`directed3`); else B0. *Motivation:* mean
  aggregation + symmetrized sampling make degree/direction non-representable, so inject
  them. *Leakage caveat:* `followers ≈ in-degree`, `statuses ≈ out-degree`, so E1 can
  win those *trivially by passthrough* — it counts as "learned structure" only if it
  beats a **raw-degree** probe baseline; the honest signal is on *content*-linked
  targets. *Hypothesis:* regression rises on structure-linked targets beyond the
  raw-degree floor; count/in-degree probes jump.

- **E2 — expressive directed aggregator** (encoder axis; *composite*). E1 + a package:
  aggregation mean→sum→**PNA**, in/out neighbors aggregated separately, readout
  mean→mean⊕sum⊕max. *Motivation:* make counts / existence / cross-neighbor
  conjunctions *representable via aggregation*, not just as static inputs.
  *Hypothesis:* count/existence probes jump, the topological signal (static-LP) widens.
  (Composite — PNA vs directed-split vs multi-readout is not individually attributed.)

- **E2b — drop-BN retry** (mechanistic follow-up, unplanned). E2 + `no_bn_encoder`.
  *Motivation:* conv BatchNorm normalizes away the *magnitude* that sum-aggregation uses
  to encode counts/degree — the named suspect for why E2's count probes stayed modest.
  *Hypothesis:* count/degree probes rise once BN is removed.

- **E4 / E4r — multi-task objective** (objective axis, built on E2's encoder). Three
  heads on one encoder: **MFR** (masked bio-feature reconstruction — the feature
  capability), **directed-LP** (score the episode's directed edges vs sampled negatives
  on the encoder embeddings — the topological task), and **structural** (reconstruct a
  masked node's `directed3` degree block from context; the node's *own* degree input is
  zeroed, so predicting it is non-trivial → no passthrough leakage). E3 (masked-feature
  reconstruction only) is folded in as the MFR head. Two combination modes: **E4** =
  simultaneous weighted sum every step; **E4r** = per-episode rotation of the heads.
  *Motivation:* make topology *used* by putting it in the **loss**, not just the
  architecture — the design's one candidate to hold the feature capability (MFR) and the
  topological capability (LP) at once. *Hypothesis:* the only arm that clears the joint
  bar. *Result:* it does not (§4, §5).

Reading chain: **B0→B1** attributes the augmentation lever; **B0→E1→E2** attributes the
encoder pieces (all under NM); **E2→E4** attributes the objective. The joint criterion is
`min(feature_score, topological_score)`, never the mean — an arm that lifts LP by
dropping regression (or vice versa) has *failed* the cross-task goal.

---

## 3. Experiment setup

- **Pretrain corpus (fixed, one seed):** the 3-way merged retweet graph
  (ukr_rus + covid + midterm), ~34 M nodes. One merged graph removes the
  pretrain-dataset multiplier. The three in-domain eval sources are *in* this mix, so
  the clean transfer read is on held-out datasets (twibot20, election2020); in-domain
  numbers are fit, not transfer.
- **Episode sampling:** within-source, source-balanced (removes the cross-source
  shortcut, equalizes exposure).
- **Budget locked at 40k episodes.** A prior budget sweep showed NM **anti-scales on
  regression**: E1 peaks at 40k (Spearman 0.222) and *degrades* by 110k (0.142); B0
  peaks ~40–60k; classification is flat across budget
  (`scripts/plotting/topology_feature_ssl/data/budget_sweep.csv`). All arms compared at
  a **true** `state_dict_40000` checkpoint (E2/E2b/E4/E4r use `epochs:5` per the trainer
  off-by-one).
- **Eval (frozen encoder):** node classification (ROC-AUC, 10-shot), node regression
  (Spearman, 10-shot), static link prediction (ROC-AUC, 0-shot). B0–E2 regression covers
  the 6-target panel; E2b/E4/E4r cover the 3 structure/age targets (followers, statuses,
  account_age). E4/E4r evals reuse E2's encoder config (`directed3` + `sage_multi`); the
  extra E4 heads are ignored on load (`strict=False`).
- **Trivial floors — an arm "improves performance" only if it beats these:**
  - `raw_feat` — linear probe of raw bio features onto each target, **no encoder**
    (`node_regression/data/features_only_floor.csv`; classification floor in
    `topology_feature_ssl/data/trivial_baselines_small.csv`).
  - `raw_degree` — linear probe of `[in_deg, out_deg, log_deg]` onto each target, the
    leakage control for E1/E2 (`topology_feature_ssl/data/leakage_baseline.csv`).
- **Free diagnostics:** planted single-rule capability probes (count-threshold,
  existence, in/out-degree, 2-neighbor conjunction), linear-probed from the frozen rep.

> **Deviations from the plan, stated plainly.** (a) **E4/E4r were run 2026-07-13**; E3
> (masked-feature-only) is folded into E4's MFR head. E4's regression eval covers the 3
> structure/age targets, so E4's reg mean is over those 3. (b) **E4 loss balancing:** the
> structural target (z-scored `directed3` degree) is power-law — a masked hub's z-score
> can be ~100, so its reconstruction MSE exploded to ~9500 and swamped MFR/LP. Fixed by
> clipping the target to ±4σ and setting `e4_weights = 30,1,1`; even so the struct term
> stayed dominant (§5). (c) The 2×2 ablation was **uninformative** for these arms
> (near-zero feature-task denominators make the retained-fraction explode/invert), so the
> topological read rests on static-LP + probes; it was not re-run for E4. (d) The
> full-panel `raw_feat`/`raw_degree` baselines rendered on the compute node were not
> synced locally; the local floors cited give an identical qualitative verdict.

---

## 4. Results — raw data

**All raw per-run rows:**
`scripts/plotting/node_regression/data/node_regression.csv`,
`scripts/plotting/node_classification/data/node_classification.csv`,
`scripts/plotting/static_link_prediction/data/static_link_prediction.csv`,
`scripts/plotting/topology_feature_ssl/data/{capability_probes_40k,budget_sweep}.csv`.
Filter to `split == "test"` and the `*_40k` arms. Consolidated workbook:
`topology_feature_ssl_results.xlsx`.

### 4.1 Regression — Spearman (test, mean over 4 datasets, 10-shot)
`node_regression.csv` · floors: `features_only_floor.csv`, `leakage_baseline.csv`

| arm | followers | friends | statuses | favourites | listed | account_age |
|---|---|---|---|---|---|---|
| _raw_degree_ (leak) | 0.162 | — | 0.156 | — | — | 0.010 |
| _raw_feat_ (floor)  | 0.187 | 0.133 | 0.089 | 0.067 | 0.144 | 0.024 |
| B0  |  0.033 | −0.072 |  0.021 | −0.002 | −0.001 | −0.062 |
| B1  | −0.127 | −0.154 | −0.107 | −0.110 | −0.109 | −0.133 |
| **E1**  | **0.191** | **0.151** |  0.095 |  0.044 |  0.141 | **0.118** |
| E2  | −0.068 | −0.105 | −0.095 | −0.078 | −0.045 | −0.069 |
| E2b | −0.041 | — | −0.009 | — | — |  0.047 |
| E4  | −0.181 | — | −0.139 | — | — | −0.079 |
| E4r | −0.233 | — | −0.133 | — | — | −0.005 |

E1 is the only arm above zero everywhere. Its unambiguous, leakage-free win is
**account_age** (0.118 vs feature-floor 0.024). **E4 and E4r are negative on all three
swept targets** (mean −0.13 / −0.12) — the multi-task objective *loses* the feature
capability, landing below even the no-op control B0.

### 4.2 Classification — ROC-AUC (test, 10-shot)
`node_classification.csv` · floor: `trivial_baselines_small.csv`

| arm | election2020 | twibot20 |
|---|---|---|
| _raw_feat_ | 0.848 | 0.560 |
| B0  | 0.981 | 0.605 |
| B1  | 0.984 | 0.613 |
| E1  | 0.953 | 0.604 |
| E2  | 0.972 | 0.589 |
| E2b | 0.969 | 0.599 |
| E4  | 0.511 | 0.378 |
| E4r | 0.651 | 0.636 |

The NM arms (B0–E2b) all clear the feature floor and cluster ~0.78. **E4 crashes below
the floor** (election2020 0.51 ≈ chance, twibot20 0.38 < chance); E4r is also degraded
(0.64). The multi-task objective breaks the feature representation classification relies on.

### 4.3 Static link prediction — ROC-AUC (test, 0-shot) — the direct topological task
`static_link_prediction.csv`

| arm | covid | midterm | twibot20 | ukr_rus | **MEAN** |
|---|---|---|---|---|---|
| B0  | 0.657 | 0.658 | 0.635 | 0.753 | 0.675 |
| B1  | 0.341 | 0.378 | 0.339 | 0.306 | **0.341** |
| E1  | 0.628 | 0.635 | 0.714 | 0.650 | 0.657 |
| **E2**  | **0.780** | **0.708** | **0.735** | **0.823** | **0.761** |
| E2b | 0.402 | 0.361 | 0.517 | 0.323 | 0.401 |
| E4  | 0.646 | 0.608 | 0.732 | 0.664 | 0.662 |
| E4r | 0.212 | 0.281 | 0.276 | 0.168 | 0.234 |

**E2 is best on every dataset (mean 0.761).** E4's directed-LP head buys only 0.662 —
**below E2 and below the do-nothing control B0 (0.675)**. E4r collapses below chance
(0.234). B1 and E2b also below chance. Putting link prediction *in the loss* (E4) did not
beat clean NM with a count-aware encoder (E2).

### 4.4 Capability probes — linear-probe AUC (chance = 0.50)
`capability_probes_40k.csv`

| arm | count_thr | in_deg | out_deg | existence | conjunction |
|---|---|---|---|---|---|
| B0  | 0.478 | 0.515 | 0.524 | 0.515 | 0.513 |
| B1  | 0.527 | 0.509 | 0.523 | 0.525 | 0.526 |
| E1  | **0.672** | **0.627** | 0.515 | 0.535 | 0.534 |
| E2  | 0.589 | 0.513 | 0.583 | **0.623** | **0.626** |
| E2b | 0.659 | 0.558 | **0.710** | 0.548 | 0.574 |
| E4  | 0.245 | 0.291 | 0.359 | 0.508 | 0.468 |
| E4r | 0.178 | 0.148 | 0.369 | 0.449 | 0.377 |

E1 leads count/in-degree (passthrough of its degree *inputs*); E2 leads the multi-neighbor
rules (existence, conjunction) that require summation. **E4/E4r sit BELOW chance on the
structural rules** (0.15–0.51) — the multi-task representation does not even *linearly
encode* the planted structure, consistent with its crashed downstream tasks. (Encoder
loaded cleanly — 0 size-mismatch — so this reflects a genuinely degraded rep, not a load
artifact.)

### 4.5 Joint bar — `min(feature, topological)` per arm
Feature = classification AUC (mean over the 2 held-out datasets); topological = static-LP
AUC (mean over 4). Regression shown for reference. Chance = 0.50 on both axes.

| arm | reg ρ | cls AUC | static-LP AUC | **min(cls, slp)** |
|---|---|---|---|---|
| B0  | −0.00 | 0.793 | 0.675 | 0.675 |
| B1  | −0.12 | 0.799 | 0.341 | 0.341 |
| E1  |  0.14 | 0.778 | 0.657 | 0.657 |
| **E2**  | −0.08 | 0.781 | **0.761** | **0.761** |
| E2b | −0.00 | 0.784 | 0.401 | 0.401 |
| E4  | −0.13 | 0.445 | 0.662 | 0.445 |
| E4r | −0.12 | 0.643 | 0.234 | 0.234 |

E2 has the highest joint bar (0.761) but negative regression; no arm is strong on **all**
of {regression, classification, static-LP}. E4/E4r — the arms designed to be — are the
**worst** on the joint bar.

---

## 5. Findings / discussion

Evidence-based headlines (each tied to a table above):

1. **No single encoder clears the joint bar — on any axis.** Encoder axis: feature-strong
   (E1: regression) and topology-strong (E2: static-LP) are *different* arms. Objective
   axis: E4/E4r are strong on *neither*. `min(feature, topological)` is low for all. The
   central hypothesis fails. *(4.1 + 4.3 + 4.5)*

2. **Count-aware aggregation, not structural inputs and not the objective, buys the
   topological task.** E2 (0.761) ≫ E1 (0.657) ≈ B0 (0.675) ≈ E4 (0.662) on static-LP.
   Aggregating over neighbors (E2) beats both injecting degree as inputs (E1) and putting
   LP in the loss (E4). *(4.3 + 4.4)*

3. **Injected degree is the only thing that makes NM reps useful for regression, and the
   win is partly genuine.** E1 is the sole arm positive across targets; its `account_age`
   gain (0.118 vs 0.024) is on a content-linked target degree passthrough cannot explain.
   *(4.1)*

4. **Cheap feature-shuffle augmentation backfired.** B1 is negative on all regression
   targets and below chance on static-LP (0.341) — the "corrupt the shortcut" lever
   destroyed usable structure. *(4.1 + 4.3)*

5. **Representable ≠ usable (the E2b lesson).** Dropping conv BatchNorm makes count/
   out-degree linearly readable (out-deg 0.52→0.71) yet crashes static-LP to 0.401. A
   probe reading a quantity out of the representation does not imply the tasks can use it.
   *(4.3 + 4.4)*

6. **The multi-task objective (E4) degrades the encoder instead of unifying it.** E4/E4r
   are negative on regression, crash classification below the floor (E4 twibot20 0.38 <
   chance), and their directed-LP head does not beat clean NM on static-LP (0.66 ≤ E2/B0;
   E4r 0.23). Joint `min(cls, slp)` = 0.45 / 0.23 — worse than every prior arm.
   **Mechanism:** even after clipping the power-law structural target, the structural-
   reconstruction term dominated the loss (weighted struct 1.5 ≫ lp 0.13 ≫ mfr 0.02), so
   the encoder over-fit degree; the near-zero MFR gradient (GTE features are trivially
   reconstructed, ~0.02 MSE at init) could not preserve feature content. *(4.1–4.5)*

7. **Classification is no longer "saturated" once the objective breaks — it just wasn't
   *stressed* by the encoder/aug arms.** B0–E2b cluster ~0.78 (a shared feature rep), but
   E4 shows the feature capability *can* be destroyed. Classification discriminates a
   *broken* encoder even where it couldn't separate the NM variants. *(4.2)*

### Where this lands

All three levers are now spent, and **none clears the joint goal** — one frozen encoder
strong on both feature and topological tasks:

- **Architecture (E1, E2)** makes topology *representable* but the capabilities separate
  by arm — degree-as-input helps features, count-aware aggregation helps topology, never
  both.
- **Augmentation (B1)** made things worse.
- **Objective (E4)** — the candidate to make topology *used* jointly with features — not
  only fails to unify them but *degrades* the encoder below the NM control.

The E4 failure has an honest caveat and an honest lesson. *Caveat:* the struct term stayed
dominant even after rebalancing, so E4 is not a perfectly-balanced test — but the fix is
not obvious, because MFR's gradient is near-zero (features are trivially reconstructed), so
simply up-weighting it amplifies noise rather than preserving content. That near-zero MFR
signal is itself the *lesson*: masked-feature reconstruction is too weak a feature anchor
to survive co-training with a structural objective — the same weakness the free-preview
flagged (fp ⊀ nm on regression). A multi-task objective that unifies the two capabilities
would need a **stronger feature-preservation objective than MFR** (or explicit
loss-balancing / uncertainty weighting), and even then the encoder-axis evidence (features
and topology trade off arm-by-arm) suggests the joint goal may require a **different
backbone**, not just a different objective. On this stack, with these levers, one frozen
encoder for both tasks was not achievable.

---

_Prior write-up archived at `FINDINGS_v1_archived.md`. Glance tables:
`RESULTS_matched40k.md` (matched-40k) and `RESULTS.md` (earlier B0/B1/E1 render).
E4 run + ops: memory `tfssl-e4-run`._
