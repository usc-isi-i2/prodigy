# AGENTS.md

## Project Context

- This repo is for graph foundation model experiments on social-media graph datasets.
- Most training and evaluation work runs on Tucker/HPC nodes, not only on the local laptop.
- Current experiment state, logs, and data all live under `/dataMeR1`.
- Prefer `/dataMeR1/...` paths for current experiment state, logs, and data.
- Do not assume `/dataMeR1/...` or `/scratch1/...` paths are mounted locally.

## Environment

- Use the `prodigy` conda environment for training and evaluation commands.
- Before running Python experiment scripts on Tucker:

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate prodigy
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
```

## Important Paths

- Training configs: `scripts/experiments/`
- COVID/UKR merged experiments: `scripts/experiments/covid_ukr/`
- Eval runner: `scripts/experiments/eval/eval_ckpts_all_graph_tasks_tucker.py`
- Analysis/export scripts: `scripts/analysis/`
- Plotting notebooks/results: `scripts/plotting/`
- Tucker repo path: `/dataMeR1/phil/gfm/prodigy`
- Tucker data root: `/dataMeR1/phil/data`

## Experiment Conventions

- Tasks:
  - `nm` = `neighbor_matching`
  - `lp` = `temporal_link_prediction`
  - `pl` = `classification`
- Checkpoints usually live under `state/<run_name>/checkpoint/state_dict_<step>.ckpt`.
- Eval logs usually live under `log/eval_<model>_to_<dataset>_<task>_<shots>shot_<timestamp>/`.
- For checkpoint trajectories, evaluate every `state_dict_<step>.ckpt` in numeric step order.

## Working Style

- Prefer existing scripts and configs over inventing new command patterns.
- Use dry runs when creating large eval sweeps.
- When adding eval helpers, make them overrideable with CLI args or environment variables.
- If a path only exists on Tucker, say that clearly instead of trying to validate it locally.
