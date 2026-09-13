# Agent Instructions

This is the canonical instruction file for AI coding agents in this repository.
Claude reads it through `CLAUDE.md`; Codex reads it directly. Favor truth over
agreement, and state uncertainty rather than guessing.

## Project Context

- This repository studies transfer of graph-foundation-model representations
  across social-media graphs and tasks, with bot detection as a held-out benchmark.
- Heavy training, evaluation, graph construction, and embedding generation run on
  Tucker. The laptop is for code edits, notebooks, plots, and lightweight checks.
- Current experiment state, logs, and data live under `/dataMeR1` on Tucker. Do not
  assume `/dataMeR1` or `/scratch1` is mounted locally.
- The Tucker repository is `/dataMeR1/phil/gfm/prodigy`; its data root is
  `/dataMeR1/phil/data`.

## Task Guides

Read the relevant guide before acting. Repository skills are shared under
`.agents/skills/` and are discovered automatically by Codex.

| Task | Required guide |
| --- | --- |
| Train, evaluate, construct graphs, generate features, or operate on Tucker | `.agents/skills/prodigy-tucker/SKILL.md` |
| Create or analyze an experiment, parse logs, or add an eval helper | `.agents/skills/prodigy-experiments/SKILL.md` |
| Add, remove, move, rename, or inventory a graph | `.agents/skills/prodigy-graph-catalog/SKILL.md` |
| Change the model, sampler, episode format, dataloader, or objective plumbing | `docs/agent_guides/model_architecture.md` |
| Move tracked evidence, merge result CSVs, or interpret legacy results | `docs/agent_guides/repo_traps.md` |
| Locate code, analyses, archives, or paper materials | `docs/agent_guides/repository_map.md` |

## Non-Negotiable Safety Rules

### Shared Git work

- Assume other agents are editing and committing concurrently. Re-run `git status`
  and `git log --oneline -1` immediately before every commit.
- Stage explicit paths only; never use `git add -A` in this working copy.
- Pull before pushing. If rejected, rebase on the remote branch and retry. Never
  force-push a shared branch.
- Preserve unrelated changes. Report the branch and local/Tucker worktree used.
- Move source code between laptop and Tucker through Git; do not hand-copy it unless
  explicitly requested.

### Tucker

- Only GPUs 0, 1, 2, and 3 are owned. Leave GPUs 4-7 untouched.
- Give every heavy or long-running experiment a dedicated Tucker worktree and tmux
  session. The user generally launches large or long-running jobs.
- Never `git checkout` or `git pull` in a worktree with a running job. Check tmux
  first, and always name the branch in a Tucker pull: `git pull origin <branch>`.
- `state/` and `log/` are gitignored and local to each worktree. Evaluate in the
  training worktree or use absolute checkpoint/log paths.
- Read-only inspection on Tucker is allowed. If SSH stalls, check USC VPN or USC
  Wi-Fi before diagnosing the host.

### Evidence and evaluation

- `docs/graph_catalog.json` is the sole graph registry. Update it first when a graph
  changes; do not create a second hard-coded complete registry.
- Use `scripts/eval/pair_link_eval.py` for static link prediction. Episodic static-LP
  results produced before 2026-07-23 are invalid; temporal LP has the same unresolved
  defect. See `docs/agent_guides/repo_traps.md` before citing either.
- Runs from before 2026-07-26 may end one checkpoint interval before their label.
  Pin comparison steps explicitly instead of comparing each run's highest checkpoint.
- Changing `--seed` does not resample evaluation episodes; it changes label
  downsampling only. Do not present seed sweeps as episode-sampling confidence.
- The shared parser skips `--ablate-features` runs. Read their metric JSON files
  directly; do not widen the parser regex without adding the ablation to its dedup key.

## Important Paths

- Training engine: `experiments/`
- Model code: `models/`
- Dataset and batching code: `data/`
- Experiment setup: `scripts/experiments/setup/<name>/`
- Experiment analysis: `scripts/experiments/analysis/<area>/.../<name>/`
- Canonical analysis index: `scripts/experiments/analysis/README.md`
- Shared train/eval/export harness: `scripts/harness/`
- Main checkpoint evaluator: `scripts/eval/eval_ckpts_all_graph_tasks_tucker.py`
- Graph registry: `docs/graph_catalog.json`
- Paper planning, outside Git: `/Users/philipp/projects/gfm/paper`

## Local Environment

- Use `/opt/homebrew/bin/python3.11` for local plots and notebooks.
- Use the local `prodigy` environment for lightweight model checks; do not train
  locally. Avoid other local conda environments unless requested.
- LibreOffice is not installed locally.

## Working Style

- Prefer existing scripts, configs, and harnesses over new one-off command patterns.
- Use dry runs for large sweeps.
- Make new evaluation helpers configurable through arguments or environment variables.
- If a path exists only on Tucker, say so rather than pretending to validate it locally.
- Keep experiment production and downstream interpretation in their separate setup and
  analysis trees. Commit evidence only in the repository's designated data/figure paths.
