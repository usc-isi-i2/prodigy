# pretrain_probe_matrix — Findings

**Date:** 2026-07-09 (matrix); 2026-07-10 (`features_only` floor completed → matrix
closed). **Scope:** frozen-probe cross-task matrix (Version B), single seed. Rows =
pretraining strategy (frozen encoder), cols = downstream task, measured **against
floors** (the new part). See `README.md` for the plan/hypothesis.

## Setup
Frozen-encoder eval (`--eval_only`), no fine-tuning. Strategy rows are existing
checkpoints: `task_transfer_covid_{nm,cl,fp}` (same covid graph, different SSL
objective) + `muc10k` (neighbor-matching on the merged ukr+covid graph, 10k). Two
floors: `random_init` (untrained encoder, same arch — routed via an empty
`--pretrained_model_run`) and `features_only` (raw bio embeddings → target,
shot-matched Ridge). Headline tasks: node regression (10-shot Spearman ρ, 6-target
panel × 4 datasets) and static link prediction (0-shot ROC-AUC, 5 datasets).

## Headline
**Two clean results, now that both floors are in.** (1) **Static-LP:** only
single-source NM learns transferable structure — +0.23 ROC-AUC over an untrained
encoder and the sole above-chance model; every other objective is indistinguishable
from random init. (2) **Regression:** the `features_only` floor (raw bio embeddings →
target) reaches ρ=**0.11**, above the random encoder (0.02) and above **all four**
pretrained encoders (ρ ≤ 0). So pretraining doesn't merely fail to add profile-rank
signal — passing features through the GNN *discards* signal that a plain ridge on the
raw embeddings keeps.

## Matrix (means over datasets; Δ = row − random_init floor)

| row (frozen encoder) | node-reg ρ | Δ reg | static-LP AUC | Δ LP |
|----------------------|-----------:|------:|--------------:|-----:|
| `NM · covid`         | −0.05      | −0.07 | **0.61**      | **+0.23** |
| `CL · covid`         | −0.00      | −0.02 | 0.33          | −0.05 |
| `FP · covid`         | −0.07      | −0.09 | 0.36          | −0.02 |
| `NM · merged` (muc10k) | −0.03    | −0.05 | 0.35          | −0.03 |
| `random_init` (floor) | 0.02      |  —    | 0.38          |  —   |
| `features_only` (floor) | **0.11** | **+0.09** | n/a       |  —   |

`features_only` reg ρ is the mean of 23 rows (4 datasets × 6 targets, twibot20 has no
`favourites_count`); positive on **4/4** datasets (covid +0.07, midterm +0.12,
twibot20 +0.12, ukr +0.12). It is a *node-target* floor, so `n/a` on static-LP. Its Δ
column is margin over the **random-encoder** floor (raw ridge beats the untrained GNN
by +0.09).

## Findings
1. **H1 holds only for NM on static-LP.** NM·covid clears the `random_init` floor by
   **+0.23** and is the **only model above chance** (0.50) on static-LP. The win is
   consistent across datasets, not a one-dataset artifact: NM·covid beats random init
   on **5/5** static-LP datasets (per-dataset Δ +0.13…+0.41) and clears chance on
   **5/5** (0.575…0.663). On node regression, **no** objective beats even the untrained
   encoder → H1 fails there.
2. **Contrastive / masked-FP / merged-NM buy nothing** on static-LP: all within
   ~0.02–0.05 of the random floor (i.e. indistinguishable from random init). Not
   "worse than random" — just no signal.
3. **Regression: encoders lose to raw features (hole filled).** The `features_only`
   floor pins the missing reference: raw bio embeddings → target under the *same*
   10-shot episodic Ridge reach ρ=**0.11** (positive on 4/4 datasets), above the random
   encoder (0.02) and above all four pretrained encoders (Δ −0.02…−0.09, ρ ≤ 0). This
   disambiguates the earlier gap — it is **"encoders lose to raw features," not "no
   headroom"**: the signal exists in the raw features, and every encoder (trained *or*
   untrained) degrades it. Consistent with the feature-content finding (NM leans on
   feature content, not topology) and NM's anti-scaling on regression (see
   `../topology_feature_ssl/`).
4. **The random-encoder floor is below chance on static-LP (0.38).** An untrained
   encoder anti-ranks true edges vs 2-hop hard negatives, so 0.38 (not 0.50) is the
   honest reference; NM at 0.61 is the only model that clears both.

## Caveats
- **Single seed — by design (checked).** The eval's episodic sampler is seeded *per
  split*, not per run: `seed = sum(ord(c) for c in split)` in every static-LP dataloader
  (`covid19_twitter.py:175`, `ukr_rus_twitter.py:176`, `midterm.py:862`,
  `social_llm_dataset.py:186`) feeds `BatchSampler(random.Random(seed))`, so `--seed`
  does **not** resample episodes — a genuine multi-seed CI would require a shared-loader
  change to reseed. Robustness instead comes from cross-dataset agreement (NM·covid
  beats random on 5/5 static-LP datasets; raw features beat the random encoder on 4/4
  regression datasets), and each cell already accumulates ~500 test episodes.
- `muc10k` static-LP is compared at **0-shot** (apples-to-apples with the others); it
  reaches ~0.85 at 10-shot, a different eval regime.
- `features_only` uses the raw episodic-Ridge floor and samples its support pool from
  all labeled nodes (vs the frozen-rep eval's test-split pool); both are the same
  10-shot/12-query/log1p protocol, so it's the intended fair reference, but it is a
  reference *ceiling* for raw features rather than a byte-identical episode set.

## Wiring (implemented — differs from the plan's TODOs)
- **`random_init` needs no checkpoint dump.** `trainer.py` only loads when
  `pretrained_model_run != ""`, so a ckpt sentinel (`NONE`) in the eval runner routes
  to an empty arg ⇒ untrained encoder. Model list: `random_init_model_list.txt`.
- **`features_only` reuses `leakage_baseline.py --features raw`** — raw bio embeddings
  through the shot-matched episodic-Ridge protocol. The raw path was made memory-safe
  (index the labeled rows out of the mmap'd feature tensor per target) so covid's
  ~23M-node graph completes without materializing a ~70GB float32 matrix. Output is its
  own file, `scripts/experiments/analysis/node_regression/data/features_only_floor.csv` (23 rows),
  read directly by the plot — *not* appended to the shared regression CSV.
- Both floor passes ran in an **isolated git worktree** off the pushed commit (the main
  Tucker tree was dirty on a different commit): `random_init` at `8101b53`,
  `features_only` at `7826419`. Figures via
  `scripts/experiments/analysis/pretrain_probe_matrix/plot_probe_matrix.py`.

## Next
Matrix is **complete** — both floors filled, both headline tasks closed. Multi-seed CIs
are descoped (episode seed is fixed per split; cross-dataset agreement stands in as the
robustness check). Feeds the weekly cross-task slide (rows = objective, cols = task).
