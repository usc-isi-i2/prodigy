# Pretrain saturation — analysis

**Status: complete, with the three follow-up checks done (2026-07-29).** 216/216 eval jobs
finished 2026-07-27, zero failures; plus a feature ablation, a run-to-run error bar, and a
probe α sweep. **Read [`FINDINGS.md`](FINDINGS.md) first**, including §5 for what the
evidence still cannot support.

## Question

Does downstream transfer performance saturate early in pretraining, and does the corpus
width change when it does?

## Inputs

Two setup folders produce one 18-point curve, keyed identically:

- [`setup/pretrain_saturation_existing/`](../../../../../../setup/pretrain_saturation_existing/) —
  steps 1000, 2000, 10000, 40000, from surviving trajectories. No training.
- [`setup/pretrain_saturation_dense/`](../../../../../../setup/pretrain_saturation_dense/) —
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
4. **Error bars now exist for classification** (`data/classification_replicates.csv`):
   two independent runs of the identical config differ by mean |Δ| 0.0122 across 48 cells.
   The step-500 rise is 16× that; the plateau is 1.1×. Use this ruler, not intuition, when
   calling a difference real — `twibot20` in particular is the noisiest graph.
5. **The random-init floor is a separate 12-job control, not a row in the curve**
   (`data/random_init_floor.csv`). It bounds the untrained encoder on a subset of cells,
   so "+0.50 over an untrained encoder" in FINDINGS §2 is measured; but "% of eventual
   gain by step 500" in §1 is computed against each arm's own best checkpoint, not against
   that floor. Do not mix the two denominators.

## Deliverables

Built by [`build_tables_and_figure.py`](build_tables_and_figure.py) (Homebrew Python 3.11
— pandas/matplotlib; the conda env is for training):

- `data/pretrain_saturation_long.csv` — 216 rows, one per (arm, step, task, dataset,
  target, metric).
- `data/pretrain_saturation_wide.csv` — arm × step, mean of the primary metric per task.
- `figures/pretrain_saturation.png` — metric vs. pretraining step, log x, one line per
  arm, one panel per task (two panels rather than two y-axes, since ROC-AUC and Spearman
  do not share a scale).
- `figures/heatmap_{classification,regression}.png` — every (step × test graph) cell, one
  panel per arm, from [`build_heatmaps.py`](build_heatmaps.py). These are where the result
  actually lives: the line figure plots the mean over test graphs, which is exactly what
  hides the per-graph split.
- `data/random_init_floor.csv` — an untrained encoder on the same cells. The control that
  separates the real classification effect from the empty regression one.
- `figures/probe_regression_{curves,heatmap}.png` — the RE-SCORED regression channel, from
  [`build_probe_figures.py`](build_probe_figures.py) reading `data/reg_probe/`. **The
  regression panels of the two figures above are void** (they plot the episodic eval whose
  `regression_head` is random and never fitted) and are titled as such; they are kept only
  as evidence of that noise. Valid regression numbers are here.
- `data/reg_probe/*.csv` — 152 rows, fitted frozen-encoder ridge probe + raw-feature floor.
- `data/step0_anchor.csv` — the untrained (`state_dict_0`) encoder on all 12 cells. All
  three arms share ONE t=0 (byte-identical, md5 `61adf822…`), so it is a single reference
  level, and it cannot sit on a log x-axis — hence a horizontal line in the curve figures
  and a real `0` column in the heatmaps.
- [`FINDINGS.md`](FINDINGS.md).

## Result in one line

Classification transfer is ~99 % complete by step 500 for the 8-source corpus and flat for
the remaining 80× of training — the rise is **16×** the measured run-to-run noise, the
plateau **1.1×** it. But the mean is carried entirely by two of the four graphs, one of the
other two sits at chance throughout, and the fourth gets *worse* with pretraining (only
~1.8σ on its own noise, so: suggestive). On the two that work the effect is large versus an
untrained encoder (+0.50/+0.58) — **but zeroing node features drops both to chance, so it
is the bio-text features doing the work, not graph structure.** There is no raw-feature
classification floor yet, so "the encoder beats the features" remains unproven. The old regression eval measured nothing (a random-init encoder matched it: the
`regression_head` is never in any checkpoint and never fitted). Re-scored with a fitted
frozen-encoder ridge probe, the channel does work — beating the raw-feature floor on 6 of
8 cells — and the effect of pretraining differs by target: `account_age_days` rises (12/12 series,
+179 %, but on a base so small the rise is only ~2σ) while `followers_count` is
flat-to-declining. Read FINDINGS §4b's conditioning caveat before quoting either: the
ridge fits on encoder embeddings are numerically degenerate (median −R² 155–725) where the
raw-feature floor's are not.
