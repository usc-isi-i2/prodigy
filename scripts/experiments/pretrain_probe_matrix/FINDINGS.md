# pretrain_probe_matrix — Findings

**Date:** 2026-07-09. **Scope:** frozen-probe cross-task matrix (Version B), single
seed. Rows = pretraining strategy (frozen encoder), cols = downstream task, measured
**against floors** (the new part). See `README.md` for the plan/hypothesis.

## Setup
Frozen-encoder eval (`--eval_only`), no fine-tuning. Strategy rows are existing
checkpoints: `task_transfer_covid_{nm,cl,fp}` (same covid graph, different SSL
objective) + `muc10k` (neighbor-matching on the merged ukr+covid graph, 10k). Two
floors: `random_init` (untrained encoder, same arch — routed via an empty
`--pretrained_model_run`) and `features_only` (raw bio embeddings → target,
shot-matched Ridge). Headline tasks: node regression (10-shot Spearman ρ, 6-target
panel × 4 datasets) and static link prediction (0-shot ROC-AUC, 5 datasets).

## Headline
**Only single-source NM learns transferable structure — +0.23 ROC-AUC over an
untrained encoder on static-LP, and the sole above-chance model. Every other
objective is indistinguishable from random init on static-LP, and *no* objective
beats the random encoder on profile regression.**

## Matrix (means over datasets; Δ = row − random_init floor)

| row (frozen encoder) | node-reg ρ | Δ reg | static-LP AUC | Δ LP |
|----------------------|-----------:|------:|--------------:|-----:|
| `NM · covid`         | −0.05      | −0.07 | **0.61**      | **+0.23** |
| `CL · covid`         | −0.00      | −0.02 | 0.33          | −0.05 |
| `FP · covid`         | −0.07      | −0.09 | 0.36          | −0.02 |
| `NM · merged` (muc10k) | −0.03    | −0.05 | 0.35          | −0.03 |
| `random_init` (floor) | 0.02      |  —    | 0.38          |  —   |
| `features_only` (floor) | _pending_ | — | n/a           |  —   |

## Findings
1. **H1 holds only for NM on static-LP.** NM·covid clears the `random_init` floor by
   **+0.23** and is the **only model above chance** (0.50) on static-LP. On node
   regression, **no** objective beats even the untrained encoder → H1 fails there.
2. **Contrastive / masked-FP / merged-NM buy nothing** on static-LP: all within
   ~0.02–0.05 of the random floor (i.e. indistinguishable from random init). Not
   "worse than random" — just no signal.
3. **Regression: null result confirmed.** All four objectives sit at or below the
   random encoder (Δ −0.02…−0.09, ρ≈0). Pretraining buys no profile-rank transfer —
   consistent with the feature-content finding (NM leans on feature content, not
   topology) and NM's anti-scaling on regression (see `../topology_feature_ssl/`).
4. **The random-encoder floor is below chance on static-LP (0.38).** An untrained
   encoder anti-ranks true edges vs 2-hop hard negatives, so 0.38 (not 0.50) is the
   honest reference; NM at 0.61 is the only model that clears both.

## Caveats
- **Single seed.** The reg deltas are within noise; the one stake-worthy claim is
  **NM +0.23 on static-LP**. Multi-seed CIs needed before promoting it.
- `features_only` reg floor still computing (covid's 23M-node feature materialize is
  slow); confirmatory only — decides "no headroom" vs "encoders lose to raw features".
- `muc10k` static-LP is compared at **0-shot** (apples-to-apples with the others); it
  reaches ~0.85 at 10-shot, a different eval regime.

## Wiring (implemented — differs from the plan's TODOs)
- **`random_init` needs no checkpoint dump.** `trainer.py` only loads when
  `pretrained_model_run != ""`, so a ckpt sentinel (`NONE`) in the eval runner routes
  to an empty arg ⇒ untrained encoder. Model list: `random_init_model_list.txt`.
- **`features_only` reuses `leakage_baseline.py --features raw`** — feeds the raw bio
  embeddings through the existing shot-matched episodic-Ridge protocol.
- Run in an **isolated git worktree** at commit `8101b53` (the main Tucker tree was
  dirty); floor rows appended to the shared node_regression / static_link_prediction
  CSVs; figures via `scripts/plotting/pretrain_probe_matrix/plot_probe_matrix.py`.

## Next
`features_only` reg floor → complete the matrix; multi-seed CIs on the NM static-LP
win; this feeds the weekly cross-task slide (rows = objective, cols = task).
