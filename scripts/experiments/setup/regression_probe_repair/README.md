# Regression probe repair

**Status:** protocol written and self-tested offline, 2026-07-27. Gate not yet run.

## Why

The episodic `task_name=regression` eval cannot measure regression. Three facts, each
checkable in the tree:

1. `models/general_gnn.py:30` builds a `regression_head` (`Linear→ReLU→Linear`) whenever
   `task_name == "regression"`, and `:158` predicts with it, bypassing `decode()` — so
   the support set's label prototypes never enter the prediction.
2. That head is **absent from every NM/CL/FP checkpoint** (38 keys, none of them
   `regression_head`), and the load is `strict=False`, so it stays at its random init.
3. `--eval_only True` makes `trainer.py:1486` run one `do_eval` under `no_grad` and
   return at `:1503`. There is no optimizer step on that path. The `--epochs 12` in the
   eval command is dead.

So every reported regression number is a **fixed random projection of the frozen
embedding**. Because `run_single_experiment.py:34` seeds before the model is built and
every job passed `--seed 0`, that projection is at least identical across arms — the
comparisons are controlled, but the metric has almost no power. It sits at or below the
raw-feature floor on exactly the targets features predict best.

This is the same class of defect as the static-LP evaluator (see
`analysis/evaluation/slp_evaluator_repair/`), and it means the ~1100 rows in
`analysis/evaluation/shared_task_tables/node_regression/data/node_regression.csv` are void as a measure of
representation quality — not just the ladder's.

## What replaces it

A frozen-encoder probe: encoder weights stay frozen, and a ridge probe is **fitted on
the support set** and scored on held-out query nodes. The protocol is lifted from
`setup/topology_feature_ssl/leakage_baseline.py:74` rather than reinvented, because that
is what produced `features_only_floor.csv` — so a probe on frozen embeddings is directly
comparable to the published floor.

Per episode: `StandardScaler` + `Ridge` fit on the support rows, predict the query rows.
Predictions accumulate across episodes and Spearman is computed **once** over the pool
(scoring per-episode and averaging is a different statistic and would not be comparable).

The episode set is built once per (dataset, target) and **shared by every arm**, so all
arms see identical support and query nodes — the property the old eval lacked and that
`pair_link_sweep.py` established for static LP.

| file | role |
|---|---|
| `scripts/eval/regression_probe.py` | the protocol; `--self-test` runs offline, 12 checks |
| `scripts/eval/regression_probe_sweep.py` | graph loaded once, every arm on one shared episode set |
| `run_gate.sh` | reproduce the published floor before trusting anything |

## Known limit of the 10-shot protocol

The self-test pins this deliberately: on a **perfectly linear planted signal** with 32
dimensions, the probe recovers ρ = 1.000 at 100 shots but only **0.371 at 10 shots**. Ten
supports cannot determine 32 coefficients, so ridge shrinkage does most of the work. The
benchmark's embeddings are 256-d and its floor 768-d, both far worse conditioned.

This bounds what any 10-shot regression number here can say. It is a reason to report a
full-data probe alongside, not a reason to distrust the fix — and it is *not* the cause of
the old numbers, which were near zero because nothing was ever fitted.

## Run order

```bash
python scripts/eval/regression_probe.py --self-test
```

```bash
bash scripts/experiments/setup/regression_probe_repair/run_gate.sh
```

**The gate is a hard stop.** It scores raw features on midterm and must reproduce
`features_only_floor.csv` (followers 0.2597, statuses 0.0546, account_age 0.0398) within
0.02. A mismatch means our protocol differs from the published floor, and every
encoder-vs-floor comparison built on it would be invalid. No GPU needed.

Only after the gate passes: the full sweep over the 21 ladder arms × 4 graphs × 3 targets,
reusing `setup/nm_ladder_downstream/model_list.txt`.

## Deferred on purpose

Controls the user asked to skip for now, to be added before any writeup:

- **random-init encoder** — an untrained GNN of the same architecture. If trained arms
  do not beat it, pretraining contributes nothing. No such baseline exists anywhere in
  the repo today, and it is the single most decisive control here.
- **label permutation** on the real graphs (the self-test covers the synthetic case).
- **alpha grid** rather than a fixed 1.0 — the sweep already accepts `--alpha a,b,c`.
- **full-data probe** as the ceiling, given the 10-shot power limit above.

## Not done without asking

`node_regression.csv` is left untouched. Those rows are void as a representation measure,
but deleting them is a separate call — the precedent is to banner-mark the findings that
cite them, the way `FINDINGS_rescore.md` handled the static-LP case.
