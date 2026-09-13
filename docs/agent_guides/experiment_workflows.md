# Experiment Workflow

Use this guide when adding, running, evaluating, or interpreting a PRODIGY
experiment.

## Structure

Experiments use two independent, name-aligned trees:

- `scripts/experiments/setup/<name>/` contains configs, launch/eval scripts, and a
  `README.md` explaining reproduction. It contains no downstream interpretation.
- `scripts/experiments/analysis/<area>/.../<name>/` contains notebooks, plotting and
  table code, `RESULTS.md`/`FINDINGS.md`, and committed `data/` and `figures/`.
  Organize by research question (`transfer`, `objectives`, `graphs`, or
  `evaluation`), then by study type/model.

Keep each experiment self-contained. The two trees are independent, so do not create
an empty counterpart when one side does not exist. Match leaf names when practical.
The known exception is setup `nm_ladder_order_robustness-jul_23` versus analysis
`nm_ladder_order_robustness`.

Use `scripts/experiments/analysis/README.md` as the canonical analysis map. Prefer
the shared train/eval harness over a one-off script. Pull cluster results into a
notebook or analysis program rather than leaving loose files at repository root.

## Evidence layout

Evaluation CSVs/JSON under an analysis `data/` directory and figures under
`figures/` are intentionally committed as evidence. Other artifact locations are
ignored. The pre-commit hook rejects files larger than 25 MB; enable it once per
clone, including Tucker, with `git config core.hooksPath .githooks`.

The ignore policy is inverted for JSON, CSV, PNG, and PDF files. Read
`docs/agent_guides/repo_traps.md` before moving evidence or merging result tables.

## Task names

The full alias map is in `experiments/params.py`:

- `nm` -> `neighbor_matching`
- `cl` or `same_graph` -> `contrastive`
- `fp` or `mfp` -> `masked_feature_prediction`
- `slp` -> `static_link_prediction`
- `reg` -> `regression`
- `mix` -> `nm_fp_cl`
- `e4` -> `e4_multi`

Unknown task names pass through unmapped, so typos fail late. `lp` and `pl` are
analysis shorthand only: `lp` means temporal link prediction and `pl` means
classification. On the command line use `temporal_link_prediction` and
`classification` in full.

## Checkpoints and logs

- Total training steps are `epochs * dataset_len_cap`.
- Checkpoints normally live at
  `state/<run_name>/checkpoint/state_dict_<step>.ckpt` and are written every
  `checkpoint_step` plus a terminal checkpoint at the true step count.
- Evaluation logs normally live at
  `log/eval_<model>_to_<dataset>_<task>_<shots>shot_<timestamp>/`.
- For trajectories, sort checkpoints by numeric step and evaluate each one.
- Before comparing final rungs, read the pre-2026-07-26 checkpoint caveat in
  `docs/agent_guides/repo_traps.md` and pin the intended comparison step.

## Implementation style

- Prefer an existing config, launcher, evaluator, or harness pattern.
- Use dry runs for large evaluation sweeps.
- Make evaluation helpers overrideable with CLI arguments or environment variables.
- Distinguish smoke runs from completed experiment results.
- If a required artifact exists only on Tucker, say so and validate it there rather
  than claiming a local check.
