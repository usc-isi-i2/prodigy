# Tucker Operations

Read this guide before mutating a Tucker checkout, launching Python there, or
creating a training/evaluation run.

## Access and ownership

- Tucker requires USC VPN or USC Wi-Fi. If SSH stalls, check connectivity first.
- Use Tucker for training, evaluation, graph construction, feature/embedding
  generation, and GPU-heavy work.
- Only GPUs 0, 1, 2, and 3 are owned. Leave GPUs 4-7 untouched.
- Long jobs run in tmux. The user generally launches large or long-running jobs.
- Read-only inspection is allowed: list directories, inspect logs and metadata, and
  load graphs without modifying them.

## Checkouts and worktrees

The main checkout is `/dataMeR1/phil/gfm/prodigy`. Long experiments use sibling
worktrees such as `/dataMeR1/phil/gfm/prodigy-<short>`. Use `git worktree list` for
the current inventory; the set changes over time.

- Give each heavy or long experiment its own worktree. Running multiple experiments
  from one checkout lets a later pull change code beneath the other experiment.
- Before changing a worktree, inspect `tmux ls` and relevant processes.
- Never `git checkout` or `git pull` in a worktree with a running job. Although an
  imported trainer may survive, later eval/follow-up processes would load the new
  revision and silently mix code versions.
- On Tucker always name the branch: `git pull origin <branch>`. Configured upstreams
  are not reliable; inspect them with
  `git rev-parse --abbrev-ref --symbolic-full-name @{u}`.
- Use Git to move source code from the laptop. Do not hand-copy source files unless
  explicitly requested.
- `state/` and `log/` are gitignored and belong to the worktree that produced them.
  A matching branch does not imply that `state/<run>/` exists there. Evaluate in the
  same worktree or pass absolute paths.

## Environments

Use `prodigy` for training and evaluation. Use `bio-embeddings-v001` for graph
construction and embedding/feature generation.

Before running a Python experiment script, put conda's binary directory on `PATH`
before sourcing and activating:

```bash
export PATH="/home/mhchu/miniconda3/bin:$PATH"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate prodigy
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
```

The `PATH` export is required. Non-interactive and `bash -lc` login shells source
`~/.bash_profile`, while conda initialization lives in `~/.bashrc`. Sourcing
`conda.sh` in a wrapper does not give a child `bash <launcher>.sh` the conda shell
function. Without the export, detached jobs can create a log and then exit with
`conda: command not found` or `/etc/profile.d/conda.sh: No such file or directory`.

Put the export inside a detached tmux command so the launched shell inherits it:

```bash
tmux new-session -d -s <name> 'export PATH="/home/mhchu/miniconda3/bin:$PATH"; bash <script.sh> --device 0'
```

## High-throughput training

Read `docs/fast_training.md` before high-throughput neighbor-matching training.
`experiments/run_shared_graph.py` shares one full CPU graph between independent
source-restricted trainers. Use its dry run and a total `--worker-budget`.

Eight models completed 200 steps each on Tucker GPU 2 with four workers each at
revision `677f50c`; that is smoke validation, not a measured concurrency optimum.
For one model, 8-16 loader workers is a measured starting point. For many models,
divide a total worker budget instead of multiplying that range by every model.
Anomaly detection is off by default; `--detect_anomaly True` enables it. Exact
training-state resume still requires zero workers.

Smoke runs must be labeled as smoke validation rather than completed experiment
results.
