# Pretrain saturation — analysis

**Status: skeleton.** The sweeps have not run yet; nothing here is populated.

## Question

Does downstream transfer performance saturate early in pretraining, and does the corpus
width change when it does?

## Inputs

Two setup folders produce one 18-point curve, keyed identically:

- [`setup/pretrain_saturation_existing/`](../../setup/pretrain_saturation_existing/) —
  steps 1000, 2000, 10000, 40000, from surviving trajectories. No training.
- [`setup/pretrain_saturation_dense/`](../../setup/pretrain_saturation_dense/) —
  steps 100, 500, from three short retrains.

Three arms — `all8` (8-source merge), `ukr`, `covid` (single-source) — × six steps.
Model keys are `sat_<arm>_s<step padded to 6>`, defined once in
`setup/pretrain_saturation_existing/arms.py`. Zero-padding means a lexical sort of a CSV
is also a numeric sort of the trajectory; do not re-sort on the raw string without
parsing the step.

Results land in the shared append-only per-task CSVs written by
`scripts/harness/benchmark_tasks/parse_benchmark_eval_logs.py`:

- `analysis/node_regression/data/*.csv` — `followers_count`, `account_age_days`, 10-shot,
  log1p, Spearman
- `analysis/node_classification/data/*.csv` — 10-shot, ROC-AUC

Filter them to rows whose model key starts with `sat_`.

## Before reading any number

1. **`check_splice.py` must have passed.** The curve mixes checkpoints from today's code
   with runs from 2026-06-14. If the splice check failed, the joint between step 500 and
   step 1000 is not interpretable.
2. **Steps 1000+ are labelled one step short.** Pre-2026-07-26 checkpoints hold N+1 steps
   under the name N. Irrelevant numerically, but do not present the axis as exact.
3. **The `±` in an eval log is the spread across episodes within one eval**, not a
   confidence interval over seeds. Eval episodes are seeded by `sum(ord(c) for c in split)`
   and ignore `--seed`, so re-running with a different seed does not resample them. For
   robustness use agreement across datasets, not a spread across seeds.
4. **There is no random-init floor row** in this design, so "saturation" here means
   "stops improving", not "reaches a fraction of the total gain over an untrained
   encoder". Do not report a percentage-of-gain without adding that row.

## Deliverables (planned)

Into `data/` and `figures/`:

- `pretrain_saturation_long.csv` — one row per (arm, step, task, dataset, target, metric).
- The curve: metric vs. pretraining step, log x, one line per arm, faceted by task.
- `FINDINGS.md` — with the saturation step per arm/task, and whether broad and narrow
  corpora differ.
