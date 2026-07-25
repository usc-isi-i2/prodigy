# Multitask SSL (mixed objective) — VALID results only

Analysis-ready consolidation of everything that survived the 2026-07-23 static-LP
evaluator rescore. Use this folder for any new analysis of the mixed-objective
experiments; do **not** pull LP numbers from the older experiment folders.

## Why this folder exists

The episodic `static_link_prediction` evaluator was found to be invalid (center-blind
scoring, frozen random prototypes, degree-confounded negatives — see
[FINDINGS_rescore.md](FINDINGS_rescore.md)). That voids every sLP number in
`multitask_ssl_rotation/` and `multitask_ssl_pairs/`, including the old headline
"emergent MIX link prediction, 0.76 AUC". The rescore of the same 15 frozen
checkpoints inverts it: **NM is the best LP arm on all 5 datasets (0.757 mean,
+0.113 over heuristic floors); MIX trails everywhere; no synergy — an NM main effect
that rotation dilutes.**

Classification (10-shot, real prototypes) and regression (regression head, no
decoder) never touched the broken path and remain valid.

| task | status | valid source |
|---|---|---|
| classification | **valid** | `multitask_ssl_pairs/data/combined_all_arms.csv` (7 arms) |
| regression | **valid** | same |
| static LP | **void → rescored** | `data/pair_lp/*.csv` (15 arms + heuristic floors) |
| temporal LP | never measured | carries the same defect; do not use without the repaired evaluator |

## Files

| file | contents | provenance |
|---|---|---|
| `data/pair_lp/*__pair_lp.csv` | rescored LP, 5 datasets × (15 arms + 5 heuristic floors) × 3 negative kinds | verbatim from `exp/slp-evaluator-repair` @ `79e173a` (`scripts/experiments/analysis/slp_evaluator_repair/results/`) |
| `FINDINGS_rescore.md` | writeup of the rescore + defect list | verbatim from same commit |
| `data/cls_reg_7arms.csv` | classification + regression, 7 lattice arms (derived) | built from `../multitask_ssl_pairs/data/combined_all_arms.csv`, old sLP rows dropped |
| `data/link_prediction_valid.csv` | arm-level LP with `best_floor_auc` / `margin_vs_floor` columns, all 3 negative kinds (derived) | built from `data/pair_lp/` |
| `data/combined_valid.csv` | one tidy long table — every valid (source, model, task, dataset, metric, value) (derived) | built from both |
| `build_valid_dataset.py` | rebuilds the three derived CSVs | — |

Rebuild: `/opt/homebrew/bin/python3.11 build_valid_dataset.py` (prints a sanity table
that must match the mean-AUC column in FINDINGS_rescore.md).

## Schema notes

- **`source`** = checkpoint family: `mtr` (rotation run: NM/CL/FP/MIX, 3-way corpus),
  `mtp` (pairs run: NMCL/NMFP/CLFP), `msc_cov`/`msc_all8` (corpora replication,
  LP-rescore only — their cls/reg live in the `exp/multitask-ssl-corpora` worktree and
  are not consolidated here). `model` is the objective combination; `k`/`group` give
  lattice depth (1=single, 2=pair, 3=triple/MIX).
- **LP headline condition** is `negative_kind == degree_matched`. `random` is easier,
  `hard_2hop` is punishing by construction (a 2-hop negative shares a neighbour and
  sits in the same community block) — don't read it as "the robust number".
- `combined_valid.csv` metrics: classification → `roc_auc`, regression → `spearman`,
  link_prediction → `auc` and `margin_vs_floor` (degree-matched only). Full metric
  sets (accuracy/f1, rmse/mae/r2, AP/hits@50, permutation gates) are in the two
  per-task CSVs.
- Dataset coverage differs by task: classification {election2020, twibot20},
  regression {covid19_twitter, midterm, twibot20, ukr_rus_twitter}, LP adds
  cp_hk_twitter (at chance for every arm — known isolated graph).
- 1 seed throughout; eval-episode seeding is per-split, so cross-arm comparisons on a
  dataset are paired, cross-seed CIs are not available.

## Headline (degree-matched LP, mean over 5 datasets)

NM .757 > NMFP .738 > NMCL/MIX .680 > CL .543 > FP .499 — every NM-containing arm
clears the heuristic floors, every arm without NM sits below them. Classification
remains an NM property (~.80 for any NM-containing arm); regression a weak FP
property. The joint-generalist story must now be argued from cls+reg vs LP-dilution
trade-offs, not from emergent LP.
