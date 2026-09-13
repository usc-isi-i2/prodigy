# Repository and Evaluation Traps

Read the relevant section before moving tracked artifacts, merging experiment
branches, parsing evaluations, or interpreting historical results. These failure
modes can succeed silently.

## Inverted artifact ignore policy

`.gitignore` blanket-ignores JSON, CSV, PNG, and PDF files, then re-includes only
designated evidence locations. The current tracked patterns include:

- `scripts/**/data/**/*.{csv,tsv,json}`
- `scripts/**/figures/**/*.{png,pdf}`
- `scripts/**/archive/**/*.{csv,png,pdf}`
- `docs/assets/**/*.png`
- `docs/archive/assets/**/*.png`
- `docs/graph_catalog.json`

Moving a tracked artifact outside a re-included path can turn the move into a silent
deletion. After any move, inspect `git check-ignore -v <path>` and confirm the new
path appears in `git ls-files`. Always stage explicit paths; never use `git add -A`.

## Shared evaluation CSVs

The per-task CSVs under
`scripts/experiments/analysis/{node_classification,node_regression,static_link_prediction}/data/`
accumulate rows across experiments. A line-wise Git auto-merge can report success
while dropping one branch's rows. Merge as a set union and verify both parents, for
example with `comm -23` over sorted unique rows from each side and the result.

`scripts/harness/benchmark_tasks/parse_benchmark_eval_logs.py` normally merges into
existing files and leaves a task CSV untouched when no run directory matches.
`--overwrite` deliberately restores replacement behavior and is safe only when the
provided log root contains every required arm. The invariant is covered by
`scripts/harness/benchmark_tasks/tests/test_parse_merge.py`.

## Invalid link-prediction results

Use `scripts/eval/pair_link_eval.py` for static link prediction. The old episodic
path is invalid because its scoring is center-blind, its prototypes are frozen and
random, and its negatives are degree-confounded. Treat all static-LP numbers produced
before 2026-07-23 as void unless checked against
`scripts/experiments/analysis/objectives/multitask/multitask_ssl/FINDINGS_rescore.md`.
Some older findings retain banner-marked void numbers. Temporal LP has the same
defect and has not been rescored.

## Historical final checkpoints

Before 2026-07-26 the training loop could not save its nominal final periodic
checkpoint. A run configured for 40k steps might therefore end at
`state_dict_30000.ckpt`. The trainer now always writes a terminal checkpoint at the
true number of executed steps.

The affected multitask-SSL arms and 120k arms (`B0`, `B1`, `E1`,
`nm_covid_midterm`, `nm_transfer_matrix`, and `twibot20_transfer`) trained one
checkpoint interval short. Their within-experiment comparisons remain meaningful,
but their labels overstate the executed steps. Configs that used five epochs as a
workaround (`nm_ladder_fillin`, `nm_ladder_order_robustness`,
`nm_single_source_matrix`, E2, and E4) can now also emit a 50k checkpoint.

Pin comparison steps explicitly. Do not compare the highest checkpoint from a
pre-fix run with the highest checkpoint from a post-fix run.

## Feature-ablation parser omission

`--ablate-features` adds a tag such as `_pl_ablP_10shot_` to the log directory, but
the shared parser expects the shot count immediately after `_pl_`. Ablation jobs can
succeed and write `metrics_test_step0.json` while contributing no CSV row. Read those
JSON files directly; an existing example is
`scripts/experiments/analysis/transfer/ablations/prodigy_nm/saturation/pretrain_saturation/data/feature_ablation.csv`.

Do not merely widen the regex. The deduplication key must also include the ablation;
otherwise an ablated and ordinary run share `(model, dataset, task, shots)` and the
newest one silently replaces the other.

## Evaluation seeds

Evaluation episodes are seeded from the split name in the dataset modules, for
example `data/covid19_twitter.py`, rather than from `--seed`. The flag affects label
downsampling but does not resample episodes. This makes arms share a fixed episode
set, but it means a seed sweep is not an episode-sampling confidence interval. The
`+/-` in logs is variation across episodes within one evaluation. Use agreement
across datasets or splits as robustness evidence.
