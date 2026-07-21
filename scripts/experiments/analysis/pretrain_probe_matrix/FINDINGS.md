# Pretraining-Strategy Probe Matrix — Findings

*Frozen-encoder transfer probe. Single seed. Built ground-up from the source CSVs
(see [Results](#results)); numbers reproduce the canonical figure aggregation exactly.*

## Executive summary

We froze four self-supervised encoders and asked whether any of them produces a
representation that transfers **above a floor** to two downstream tasks with real
headroom — node regression (10-shot) and static link prediction (0-shot) — where
"floor" means an untrained encoder of the same architecture (`random_init`) and, for
regression, a ridge on the raw features with no GNN (`features_only`).

**One narrow win, one clean null.**

1. **Exactly one of four objectives clears any floor, on exactly one of two tasks.**
   Single-source **neighbor-matching (NM) on covid beats the random-encoder floor on
   zero-shot static LP by +0.23 ROC-AUC** (0.61 vs 0.38), and does so consistently on
   all five datasets (0.58–0.66). Contrastive, feature-prediction, and merged-source
   NM are all **at or below** the random-encoder floor on LP.
2. **On node regression, pretraining is strictly harmful.** Raw features reach
   ρ = 0.109; *every* frozen encoder — trained or untrained — collapses to ρ ≈ 0 or
   negative. The untrained encoder (0.022) beats all four trained objectives. Nothing
   clears either floor.

So the only defensible "best strategy" claim is narrow and mechanistic: **NM, and only
for link prediction.** For regression the pre-registered null result holds — pretraining
buys no transfer beyond raw features.

## Methodology

### Design logic

The prior fine-tuning matrix (`../covid_task_transfer_matrix`) was uninterpretable
because full adaptation drove every cell — including from-scratch — to ceiling (CL/FP
eval spreads < 1e-3). The fix here: **freeze** the encoder (`--eval_only`, no
fine-tuning) and probe downstream tasks that still have headroom, scored as
**margin-over-floor** so "best" means *best above what you get for free*, not *closest
to a saturated ceiling*.

### Hypotheses (pre-registered in README)

- **H1 (primary):** at least one objective yields a frozen representation that beats
  **both** floors on the headline tasks.
- **H2 (ranking):** objectives rank by margin-over-floor, and the ranking is
  **consistent across tasks** → that top objective is "the best strategy."
- **Null is a valid takeaway:** if nothing clears the floors, pretraining buys no
  transfer beyond raw features on these tasks.

### Treatments and the motivation behind each

All four trained encoders share one architecture (SAGE, 768-d bio-embedding input,
emb_dim 256) and differ only as noted, so differences are attributable to the
**pretraining objective / data**, not capacity.

| row | what it is | why it's in the matrix (hypothesis it tests) |
|-----|------------|-----------------------------------------------|
| **NM · covid** | neighbor-matching, covid graph, step 11k | Pretext = predict adjacency ≈ the LP task itself → *should* transfer best to link prediction if any objective does. |
| **CL · covid** | contrastive, same covid data/step | Augmentation-invariance / instance discrimination → tests whether global-structure invariances transfer without an explicit link pretext. |
| **FP · covid** | masked feature-prediction, same covid data/step | Reconstructs masked node features → most feature-aligned objective; tests whether modeling the feature manifold helps the feature-derived regression targets. |
| **NM · merged** (`muc10k`) | neighbor-matching on the **merged ukr+covid** graph, ~10k steps | More/diverse pretraining data → tests whether merging a second source improves transfer over single-source NM (the merged-vs-single question). |
| **`random_init`** (floor) | untrained encoder, identical architecture | Isolates what the architecture + episodic readout give with **zero learning** — the "for free" baseline. |
| **`features_only`** (floor) | ridge on raw 768-d bio embeddings, **no GNN** | Isolates label signal already in the raw features → is the GNN adding anything at all? (Regression targets only.) |

### Decision rule

Per headline task, Δ = strategy − max(applicable floors). H1 is supported only if some
objective has Δ > 0 on a task; H2 only if the winning objective is consistent across
tasks. Single seed → trust only **sizable, consistent** gaps; small inter-objective Δ
is treated as noise.

## Experiment setup

- **Runner:** `../eval/eval_ckpts_all_graph_tasks_tucker.py` with `--eval_only`
  (frozen encoder, no fine-tuning). `random_init` via the `NONE` ckpt sentinel →
  empty `--pretrained_model_run` → untrained weights (no checkpoint dump).
- **Seeds:** single seed (no CIs this pass).
- **Node regression** — 10-shot episodic Ridge, `log1p` target transform, metric =
  Spearman ρ. Datasets: midterm, ukr_rus_twitter, covid19_twitter, twibot20 ×
  6 targets (followers/friends/statuses/favourites/listed counts, account_age_days);
  twibot20 lacks favourites → **23 (dataset×target) cells** aggregated per row.
- **Static link prediction** — **0-shot** (zero-shot), ROC-AUC, hard negatives,
  `--slp-n-query 4`. Datasets: midterm, ukr_rus_twitter, covid19_twitter,
  cp_hk_twitter, twibot20 (**5**).
- **`features_only` floor** — `../topology_feature_ssl/leakage_baseline.py --features raw`:
  raw bio embeddings → target through the same shot-matched episodic-Ridge protocol
  (`--shots 10 --n-query 12 --episodes 500 --transform log1p`), same 23 cells.
- **Classification** (twibot20, 3-shot ROC-AUC) was run for the three covid objectives
  but **no floor row** (`random_init`/`features_only`) was evaluated for it → not
  interpretable under the decision rule; reported below but excluded from the matrix.
- **Timeline:** objective rows evaluated 06–07 Jul 2026; `random_init` floor 09 Jul;
  `features_only` floor + figures 10 Jul.

## Results

**Raw data (source of truth):**
- Regression: [`scripts/experiments/analysis/node_regression/data/node_regression.csv`](../../plotting/node_regression/data/node_regression.csv)
- Static LP: [`scripts/experiments/analysis/static_link_prediction/data/static_link_prediction.csv`](../../plotting/static_link_prediction/data/static_link_prediction.csv)
- `features_only` floor: [`scripts/experiments/analysis/node_regression/data/features_only_floor.csv`](../../plotting/node_regression/data/features_only_floor.csv)
- Classification: [`scripts/experiments/analysis/node_classification/data/node_classification.csv`](../../plotting/node_classification/data/node_classification.csv)
- Figures: [`probe_matrix_heatmap.png`](../../plotting/pretrain_probe_matrix/probe_matrix_heatmap.png),
  [`probe_matrix_delta_bars.png`](../../plotting/pretrain_probe_matrix/probe_matrix_delta_bars.png)
- Reproduce aggregation + figures: `python3 scripts/experiments/analysis/pretrain_probe_matrix/plot_probe_matrix.py`

### Node regression — 10-shot Spearman ρ (mean over 23 dataset×target cells)

| row | mean ρ | Δ vs random | Δ vs features |
|-----|-------:|------------:|--------------:|
| **`features_only`** (floor) | **0.109** | +0.087 | — |
| **`random_init`** (floor) | 0.022 | — | −0.087 |
| CL · covid | −0.002 | −0.024 | −0.111 |
| NM · merged (`muc10k`) | −0.032 | −0.054 | −0.140 |
| NM · covid | −0.053 | −0.075 | −0.161 |
| FP · covid | −0.069 | −0.091 | −0.177 |

Every trained encoder is **below both floors**; the ordering of the floors —
`features_only` ≫ `random_init` ≫ all trained encoders — is the whole story. Per-dataset,
`features_only` leads on all four datasets (ρ 0.07–0.12); no encoder is positive on more
than one dataset.

### Static link prediction — 0-shot ROC-AUC (mean over 5 datasets)

| row | mean AUC | Δ vs random |
|-----|---------:|------------:|
| **NM · covid** | **0.612** | **+0.229** |
| `random_init` (floor) | 0.382 | — |
| FP · covid | 0.362 | −0.020 |
| NM · merged (`muc10k`) | 0.352 | −0.030 |
| CL · covid | 0.334 | −0.048 |

*(`features_only` is a node-target floor — undefined for LP.)*
NM · covid is above chance (0.5) and above floor **on every dataset** (midterm 0.589,
ukr 0.644, covid 0.575, cp_hk 0.663, twibot20 0.587). Every other row — including the
untrained floor — sits **below 0.5 on all five datasets**.

### Classification — twibot20, 3-shot ROC-AUC (no floor → not in the matrix)

FP 0.673 · CL 0.607 · NM 0.595. No `random_init`/`features_only` row was evaluated, so
margin-over-floor is undefined and this column cannot discriminate strategies.

## Findings / discussion

Evidence-based headlines:

1. **H1 holds only in the narrowest sense.** Just one of four objectives clears a
   floor, on one of two tasks: **single-source NM on covid, +0.23 AUC over the
   untrained encoder on 0-shot link prediction**, robust across all five datasets. This
   is also the most mechanistically expected result — NM's pretext (predict adjacency)
   *is* the LP task — so the news is not that NM transfers to LP, but that it is the
   **only** objective that does.

2. **Pretraining is harmful, not just unhelpful, for node regression.** Raw features
   give ρ = 0.109; pushing them through *any* frozen encoder — trained or untrained —
   collapses ρ to ≈ 0, and the untrained encoder (0.022) beats all four trained ones.
   The GNN is discarding feature signal that a plain ridge keeps. → **for regression,
   use `features_only`; pretraining buys nothing.** This is the pre-registered null.

3. **H2 (consistent ranking) fails — "best strategy" is task-specific.** NM wins LP but
   is 2nd-*worst* on regression; CL is least-bad on regression but *worst* on LP. No
   objective dominates both headline tasks, so there is no single "best pretraining
   strategy" — only a best strategy *per task*.

4. **Merging sources destroys the one thing that worked.** Merged NM (ukr+covid) falls
   to 0.352 AUC on LP — below the untrained floor and far below single-source covid NM
   (0.612). Adding a second pretraining source erased the transferable adjacency
   structure. Consistent with the broader within-source > merged pattern.

5. **The trained encoders (except NM-on-LP) are indistinguishable from noise.** On LP,
   CL/FP/merged sit at or below an *untrained* encoder and below chance; on regression
   all four sit below an untrained encoder. Their pairwise differences are small and
   single-seed → do not rank them.

**Bottom line.** Of the pre-registered outcomes, we land between "one objective wins"
and "null": **NM is the best strategy for link prediction and its gain is real and
consistent, but no objective beats raw features on regression, and no objective is best
across tasks.** Frozen SSL representations transfer here only when the pretext task
structurally matches the downstream task (adjacency → link prediction); otherwise raw
features win.
