# Pretraining-Strategy Probe Matrix

**Goal:** find the best *pretraining strategy* (self-supervised objective) by how
transferable its **frozen** representation is — anchored to **floors** so "best"
means *best above what you get for free*, not *closest to a saturated ceiling*.

Extends `../pretrain_strategy_benchmark` (same frozen-probe runner) by adding two
floor rows and an explicit hypothesis + decision rule. **Single seed** this pass
(no multi-seed / CI requirement).

## Why the previous matrix wasn't interpretable

`../covid_task_transfer_matrix` fine-tuned each SSL objective back onto the *same*
COVID graph, so every cell — including the from-scratch baseline — hit ceiling:

- CL eval: ROC-AUC **0.99943 from scratch** vs 0.99950 best → 7e-5 spread.
- FP eval: MSE ≈ 0 everywhere (differences inside the score std).
- Only NM eval had any spread, and mostly in accuracy.

No headroom = no signal. Fix: **freeze** the encoder and probe *downstream* tasks
that aren't saturated, measured **against floors**.

## Hypothesis

- **H1 (primary):** at least one pretraining objective yields a frozen
  representation that beats **both** floors (random-init encoder, features-only)
  on the headline downstream tasks (node regression, static LP).
- **H2 (ranking):** the objectives rank by *margin-over-floor*, and the ranking is
  consistent across the headline tasks → that top objective is the "best strategy."
- **Null result is also a valid takeaway:** if nothing clears the floors,
  pretraining buys no transfer beyond raw features on these tasks (consistent with
  the feature-content finding — features carry most of the label signal).

## Matrix

Rows = pretraining strategy (frozen encoder). Cols = downstream eval task.

| row (strategy)  | source                                             |
|-----------------|----------------------------------------------------|
| `nm`            | neighbor-matching checkpoint                       |
| `cl`            | contrastive checkpoint                             |
| `fp`            | (masked) feature-prediction checkpoint             |
| `random_init`   | **floor** — untrained encoder, same architecture   |
| `features_only` | **floor** — logreg/ridge on raw features, no GNN   |

| col (task)         | metric            | shots        | note                          |
|--------------------|-------------------|--------------|-------------------------------|
| node regression    | Spearman (log1p)  | 10           | headline — has headroom       |
| static LP          | ROC-AUC           | 0 (zero-shot)| headline — `--slp-n-query 4`  |
| neighbor matching  | ROC-AUC           | 3            | SSL-objective col (near-ceiling; completeness) |
| classification     | ROC-AUC / F1      | 3            | twibot20 only                 |

Columns are **both** SSL-objective probes (`nm`) *and* real downstream benchmarks
(`reg`, `slp`, `classification`), per the design call.

## Protocol

- **Frozen encoder, `--eval_only`** — reuses `../eval/eval_ckpts_all_graph_tasks_tucker.py`.
  No fine-tuning; full adaptation is what erased the signal last time.
- **One seed.** Trust only sizable gaps; small Δ = inconclusive (we skipped CIs).
- **Headroom gate — run first:** evaluate the two floors on each task. If
  `random_init` or `features_only` already scores ≥ ~0.95 on a task, **drop that
  task** — it can't discriminate strategies. (Expect the `nm` column to fail this
  gate; keep it only for completeness.)

## The two floors (the new part)

1. **`features_only`** — raw node features → target, no graph structure. Reuse
   `../feature_ablation/feature_label_probe.py` for classification/nm-style labels;
   add a ridge analog for the regression target. One score per (task, dataset).
2. **`random_init`** — same encoder architecture, untrained weights. **No checkpoint
   dump needed:** `trainer.py` only loads when `pretrained_model_run != ""`, so a ckpt
   sentinel (`NONE`) in the eval runner routes to an empty arg ⇒ the encoder evals
   untrained. Row lives in `random_init_model_list.txt`.

**Wired (implemented, differs from the original TODOs):** (a) `random_init` via the
`NONE`/`""` ckpt sentinel in `../eval/eval_ckpts_all_graph_tasks_tucker.py` (no dump);
(b) `features_only` via `../topology_feature_ssl/leakage_baseline.py --features raw`
(raw bio embeddings through the existing shot-matched episodic-Ridge protocol).
**Results: see `FINDINGS.md`.**

## Run

```bash
# 0a. features_only floor (CPU): raw bio embeddings -> target, shot-matched Ridge.
#     --skip-fulldata is REQUIRED on the big graphs: the full-data Ridge reference is
#     O(n·d²) and would hang on covid's 23M nodes. The raw path indexes labeled rows
#     out of the mmap'd tensor, so peak RAM is (n_labeled × 768), not the full matrix.
python3 ../topology_feature_ssl/leakage_baseline.py --features raw \
  --data-root /dataMeR1/phil/data \
  --datasets midterm,ukr_rus_twitter,covid19_twitter,twibot20 \
  --targets followers_count,friends_count,statuses_count,favourites_count,listed_count,account_age_days \
  --shots 10 --n-query 12 --episodes 500 --transform log1p --skip-fulldata
#    -> scripts/plotting/node_regression/data/features_only_floor.csv (23 rows)

# 0b. random_init floor via the normal runner (empty ckpt = untrained encoder)
python3 ../eval/eval_ckpts_all_graph_tasks_tucker.py --tasks reg \
  --datasets midterm,ukr_rus_twitter,covid19_twitter,twibot20 --reg-transform log1p --shots 10 \
  --model-list random_init_model_list.txt --data-root /dataMeR1/phil/data --gpus 0,1
python3 ../eval/eval_ckpts_all_graph_tasks_tucker.py --tasks slp \
  --datasets midterm,ukr_rus_twitter,covid19_twitter,cp_hk_twitter,twibot20 \
  --slp-hard-negatives True --slp-n-query 4 --shots 0 \
  --model-list random_init_model_list.txt --data-root /dataMeR1/phil/data --gpus 2,3
#    strategy rows (task_transfer_covid_{nm,cl,fp}, muc10k) are already in the CSVs

# 1. collect + plot (results keyed by model = row)
python3 ../../analysis/benchmark_tasks/parse_benchmark_eval_logs.py --log-root <log-dir> --out-dir scripts/plotting
python3 ../../plotting/pretrain_probe_matrix/plot_probe_matrix.py   # heatmap + Δ-over-floor bars
```

## Read-out / takeaway template

Per headline task compute `Δ = strategy − max(random_init, features_only)`:

> "Frozen probe, 1 seed. On node-regression `<obj>` beats the features-only floor
> by Δ=`<..>` Spearman and the next objective by `<..>`; on static-LP by `<..>`.
> `nm`/`fp` sit within noise of the floor → **best pretraining strategy = `<obj>`,
> and its gain is above raw features.**"

If every Δ ≤ ~0 → "no objective beats features-only; pretraining isn't buying
transfer on these tasks."

## Reuse (don't reinvent)

- runner: `../pretrain_strategy_benchmark/run_pretrain_strategy_benchmark.sh` + `../eval/eval_ckpts_all_graph_tasks_tucker.py`
- downstream tasks: `../node_regression/`, `../static_link_prediction/`
- floors: `../feature_ablation/feature_label_probe.py`, `../feature_ablation/feature_only_nm_probe.py`
- plots: `scripts/plotting/best_pretrain_strat/` (already has headroom-normalized views)
