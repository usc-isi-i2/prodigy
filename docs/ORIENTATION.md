# Orientation for new sessions

## Compute
- Heavy work (training, eval) runs on a shared GPU cluster called **Tucker** — ssh in to use it. The repo, data, and checkpoints all live there too, under **`/dataMeR2`** (older `/dataMeR1` paths are deprecated — don't use them), with conda envs already set up.
- **Tucker conda envs**: use **`prodigy`** for training/eval (and model code generally); use **`bio-embeddings-v001`** for graph construction and embedding/feature generation.
- **We own GPUs 0–3**; the rest belong to another group. Check what's free and stay on ours. Long jobs run in tmux; the user kicks off the big/long ones.
- **Read vs. write on Tucker**: reading on Tucker (ssh in to inspect files, list dirs, check logs, load graphs read-only) is fine — go ahead. For **write ops** (launching training/eval, building artifacts, moving/deleting files), the user generally prefers to run the commands himself — hand him the exact command rather than executing it.

## Working laptop ↔ cluster
- **Use git to move code** — commit/push from the laptop, pull on the cluster. Don't hand-copy files around.
- The working branch shifts over time, and more than one agent may be active at once — check which branch you're on before committing; don't assume it's last session's.

## Experiments & results
- Keep experiments **atomized**: each one self-contained (config + runner + notes) in its own subfolder, reproducible on its own.
- Experiment code goes under the experiments area; analysis/plots go under the plotting area as notebooks. Pull results from cluster logs into a notebook rather than leaving them loose.
- Prefer the shared train/eval harness over one-off scripts.

## Local
- The laptop is only for light work (running notebooks/plots); anything GPU goes to the cluster.
- **Local conda envs**: for plotting/notebooks use the Homebrew conda Python 3.11 (**`/opt/homebrew/bin/python3.11`**, has numpy/pandas/matplotlib); for model code use the local **`prodigy`** env (mirrors Tucker) — but note we don't run any training locally, that's Tucker-only. Avoid the other local conda envs (older 3.7–3.10 / 3.13 base).

## What this project is
- Studying how graph-foundation-model representations transfer across social-media graphs and tasks — including bot detection as a held-out benchmark.
