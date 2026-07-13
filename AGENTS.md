# Agent Instructions

This is the canonical instruction file for AI coding agents in this repo.
Claude should read this via `CLAUDE.md`; Codex/GPT reads this file directly.

## Project Context

- This repo is for graph foundation model experiments on social-media graph datasets.
- Most training and evaluation work runs on Tucker/HPC nodes, not only on the local laptop.
- Current experiment state, logs, and data all live under `/dataMeR1`.
- Prefer `/dataMeR1/...` paths for current experiment state, logs, and data.
- Do not assume `/dataMeR1/...` or `/scratch1/...` paths are mounted locally.
- Heavy work runs on Tucker; the laptop is for light code edits, notebooks, and plots.
- Studying how graph-foundation-model representations transfer across social-media graphs and tasks, including bot detection as a held-out benchmark.

## Compute

- Reaching Tucker requires the USC VPN active, or being on USC wifi. If ssh to Tucker stalls, first check whether VPN is connected.
- Use Tucker for training, eval, graph construction, embedding generation, and any GPU-heavy workflow.
- We own GPUs 0-3 on Tucker; the rest belong to another group. Check availability and stay on ours.
- Long jobs run in tmux. The user generally kicks off big or long-running jobs.
- Reading on Tucker is fine: inspect files, list dirs, check logs, and load graphs read-only.
- For write operations on Tucker, such as launching training/eval, building artifacts, or moving/deleting files, prefer giving the exact command for the user to run unless they explicitly ask you to execute it.

## Laptop/Cluster Workflow

- Use git to move code between laptop and cluster: commit/push from laptop, pull on Tucker.
- Do not hand-copy source files between laptop and cluster unless explicitly requested.
- The working branch shifts over time and multiple agents may be active at once. Check branch and status before committing.

## Environment

- Use the `prodigy` conda environment for training and evaluation commands.
- On Tucker, use `bio-embeddings-v001` for graph construction and embedding/feature generation.
- Before running Python experiment scripts on Tucker, put conda's `bin` on `PATH` **first**, then source and activate:

```bash
export PATH="/home/mhchu/miniconda3/bin:$PATH"   # so the `conda` executable resolves
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate prodigy
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
```

- **Why the `PATH` export is required — common failure, read this before launching.** Non-interactive shells and `bash -lc` login shells source `~/.bash_profile`, not the `~/.bashrc` where conda-init lives, so `conda` is not on `PATH`. Sourcing `conda.sh` in a *wrapper* does **not** fix a launcher script run as `bash train_arm_tucker.sh`: the child process does not inherit the `conda` shell function. Exporting conda's `bin` onto `PATH` lets the child script run its own `conda info --base` / `conda activate`. Symptom when missing: `conda: command not found` and `/etc/profile.d/conda.sh: No such file or directory`, and detached tmux jobs that exit immediately (log created, session gone).
- When launching a heavy job in detached tmux, put the export **inside** the tmux command so the child inherits it:

```bash
tmux new-session -d -s <name> 'export PATH="/home/mhchu/miniconda3/bin:$PATH"; bash <script.sh> --device 0'
```

- For local plotting/notebooks, use the Homebrew conda Python 3.11 at `/opt/homebrew/bin/python3.11`; it has numpy/pandas/matplotlib.
- For local model-code checks, use the local `prodigy` env, but do not run training locally.
- Avoid other local conda envs unless the user explicitly asks.
- LibreOffice is not installed locally; `soffice`/`libreoffice` are unavailable for converting or rendering documents (.pptx/.xlsx/.docx/.pdf).

## Important Paths

- Training configs: `scripts/experiments/`
- COVID/UKR merged experiments: `scripts/experiments/covid_ukr/`
- Eval runner: `scripts/experiments/eval/eval_ckpts_all_graph_tasks_tucker.py`
- Analysis/export scripts: `scripts/analysis/`
- Plotting notebooks/results: `scripts/plotting/`
- Tucker repo path: `/dataMeR1/phil/gfm/prodigy`
- Tucker data root: `/dataMeR1/phil/data`

## Experiment Conventions

- Keep experiments atomized: each experiment should be self-contained, with config, runner, and notes in its own subfolder.
- Experiment code belongs under `scripts/experiments/`.
- Analysis and plots belong under `scripts/plotting/` as notebooks or plotting scripts.
- Pull results from cluster logs into a notebook rather than leaving loose result files at the repo root.
- Prefer the shared train/eval harness over one-off scripts.
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
