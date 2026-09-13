---
name: prodigy-experiments
description: Create, modify, run, evaluate, or analyze PRODIGY experiments and their configs, launchers, logs, checkpoints, findings, figures, or result tables. Use for work under scripts/experiments, scripts/eval, or scripts/harness; do not use for model internals alone.
---

# PRODIGY Experiments

Build on an existing experiment or shared harness whenever possible. Keep run
production reproducible and keep interpretation tied to committed evidence.

## Before acting

1. Read [experiment workflow](../../../docs/agent_guides/experiment_workflows.md).
2. Locate the closest existing setup and analysis through
   `scripts/experiments/analysis/README.md`; inspect those before inventing structure.
3. If the work runs on Tucker, also use the `prodigy-tucker` skill.
4. For a multi-arm ladder, matrix, ablation grid, staged selection, or other campaign,
   also use `prodigy-controlled-campaigns`.
5. Before parsing, merging, comparing, or interpreting results, also use
   `prodigy-evaluation-integrity`.

## Required outcomes

- Keep run inputs and launch/eval instructions in
  `scripts/experiments/setup/<name>/`.
- Keep notebooks, summaries, findings, result data, and figures in the matching
  `scripts/experiments/analysis/<area>/.../<name>/` leaf.
- Do not create an empty counterpart directory when only setup or analysis exists.
- Put `README.md` reproduction instructions with setup; put `RESULTS.md` or
  `FINDINGS.md` with analysis.
- Preserve shared result CSVs as append-only unions across experiments.
- Use the exact task aliases and checkpoint semantics documented in the workflow
  reference. Verify legacy checkpoint and evaluation caveats before comparisons.
- Store committed CSV/JSON evidence under `data/` and figures under `figures/` so the
  repository's inverted ignore rules do not silently drop them.
- Prefer configurable evaluation helpers and dry-run large sweeps.

When reporting results, distinguish smoke validation, completed runs, missing inputs,
and conclusions supported by actual evaluation artifacts.
