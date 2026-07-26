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

- For local plotting/notebooks, use the Homebrew Python 3.11 at `/opt/homebrew/bin/python3.11` (not a conda env); it has numpy/pandas/matplotlib.
- For local model-code checks, use the local `prodigy` env, but do not run training locally.
- Avoid other local conda envs unless the user explicitly asks.
- LibreOffice is not installed locally; `soffice`/`libreoffice` are unavailable for converting or rendering documents (.pptx/.xlsx/.docx/.pdf).

## Important Paths

- Experiment setup (configs, launch/eval scripts): `scripts/experiments/setup/<name>/`
- Experiment analysis (notebooks, findings, data, figures): `scripts/experiments/analysis/<name>/`
- COVID/UKR merged experiments: `scripts/experiments/setup/covid_ukr/`
- Eval runner: `scripts/eval/eval_ckpts_all_graph_tasks_tucker.py`
- Shared analysis/export harness: `scripts/harness/`
- Retired analyses: two distinct sets, do not conflate them.
  `scripts/experiments/analysis/archive/` holds the ones kept in the working tree;
  the 23 removed on 2026-07-26 are only in git (see *Where to Start Reading*).
  `archive/README.md` is the authoritative split.
- Paper planning: `docs/paper/` — currently `state_doc_jul22.md` and
  `related_work/`; superseded drafts in `docs/paper/archive/`. Prose planning docs
  belong here, **not** in `scripts/experiments/setup/`.
- There is no `slides/` tree. The decks and their pptxgenjs build scripts were
  removed from the repo on 2026-07-26 (`6dd1635`); recover from git if needed.
- Training engine (trainer, params, sampler): `experiments/` at the repo root —
  note this is the model code, *not* the per-experiment folders above.
- Tucker repo path: `/dataMeR1/phil/gfm/prodigy`
- Tucker data root: `/dataMeR1/phil/data`

## Graph Catalog

- `docs/graph_catalog.json` is the single source of truth (moved from `config/` on
  2026-07-26; loaders accept either location) for graph names,
  artifact paths, source composition, and evaluation capabilities.
- Use each entry's `canonical_name` in prose and new documentation. Use
  `dataset_key` where compatibility with existing configs, CLI arguments, logs,
  and loaders is required; do not rename historical artifacts retroactively.
- Resolve `relative_path` beneath the catalog's `data_root`. The only current
  Tucker data root is `/dataMeR1/phil/data`; do not introduce alternative roots.
- When adding, removing, moving, or changing a graph, update the catalog first.
  Code that needs a complete graph registry should read the catalog rather than
  defining another hard-coded list. Experiment-specific subsets remain allowed.
- Keep inventory metadata current: artifact byte size, node/edge counts, features,
  labels, supported tasks, source-data locations, metadata sidecar, and construction
  provenance. Verify these read-only on Tucker and advance the catalog's top-level
  `last_verified` (there is no per-graph field — entries carry `canonical_name`,
  `dataset_key`, `relative_path`, `statistics`, `tasks`, `construction`, …); use
  explicit `null` values or notes when a fact is unknown rather than guessing.

## Repo Traps (read before moving files or merging branches)

Learned the hard way in the 2026-07-26 consolidation. Each of these fails *silently*.

- **The ignore policy is inverted, so moving a tracked data file untracks it.**
  `.gitignore` blanket-ignores `*.json`, `*.csv`, `*.png`, `*.pdf` and re-includes only
  these paths — the list is exhaustive, anything else stays ignored:
  `scripts/**/data/**/*.{csv,tsv,json}`, `scripts/**/figures/**/*.{png,pdf}`,
  `scripts/**/archive/**/*.{csv,png,pdf}`, `docs/assets/**/*.png`,
  `docs/graph_catalog.json`. Move such a file outside a re-included path and `git add -A`
  will quietly *drop* it — the commit reads as a deletion. Note `docs/archive/` is **not**
  re-included, so archiving a figure out of `docs/assets/` untracks it. After moving any
  data file, check `git check-ignore -v <path>` and confirm it still appears in
  `git ls-files`.
- **Scope `git add`.** This working copy has held unrelated projects; an unscoped
  `git add -A` swept one into a research commit and needed a history rewrite to undo.
  Stage the paths you actually changed.
- **The shared per-task eval CSVs are append-only accumulations across experiments**
  (`analysis/{node_classification,node_regression,static_link_prediction}/data/*.csv`).
  Every experiment appends its arms. A git line-wise auto-merge of two branches that
  both appended will report success while dropping rows. Merge them as a **union** and
  verify with `comm -23 <(git show <side>:<path> | sort -u) <(sort -u <path>)` for both
  sides before committing.
- **Static link prediction: use `scripts/eval/pair_link_eval.py`.** The episodic sLP
  path in the old runner is invalid (center-blind scoring, frozen random prototypes,
  degree-confounded negatives). Every sLP number produced before 2026-07-23 is void —
  do not cite one without checking it against
  `analysis/multitask_ssl/FINDINGS_rescore.md`. Findings that still carry void numbers
  are banner-marked; temporal LP has the same defect and was never rescored.
- **Runs before 2026-07-26 are one `checkpoint_step` shorter than their label.**
  The training loop is `trange(self.steps)`, so `e` never reaches `self.steps` and the
  periodic `e % checkpoint_step == 0` save could not fire on the final step. A config
  with `epochs:4 × dataset_len_cap:10000` and `checkpoint_step:10000` — labelled 40k —
  left `state_dict_30000` as its last checkpoint, and trajectory evals stopped there.
  Fixed 2026-07-26: the trainer now always writes a terminal
  `state_dict_<steps_run>.ckpt`. Two consequences. (1) The multitask-SSL arms
  (`setup/multitask_ssl_*/{configs,cov,all8}/*.yaml`, `epochs:4`) and the 120k arms
  (`B0`/`B1`/`E1`, `nm_covid_midterm`, `nm_transfer_matrix`, `twibot20_transfer`) were
  all trained short — uniformly, so within-experiment comparisons stand, but the step
  labels do not. (2) Configs that used the `epochs:5` workaround to land a 40k final
  (`nm_ladder_fillin`, `nm_ladder_order_robustness`, `nm_single_source_matrix`, E2/E4)
  will now *also* emit `state_dict_50000`. Pin the comparison step explicitly in
  analyses; do not take "the highest-numbered checkpoint" across pre- and post-fix runs.
- **Eval episode sampling ignores `--seed`.** Episodes are seeded by
  `seed = sum(ord(c) for c in split)` in all nine dataset modules (e.g.
  `data/covid19_twitter.py:218`), so the eval episode set is a fixed function of the
  split name. `--seed` reaches only label downsampling. This is deliberate and useful —
  arms are compared on identical episodes — but it means re-running with a different
  seed does **not** resample eval episodes. Never report a spread across `--seed` values
  as an eval-episode confidence interval; the `±` in the eval logs is the std across
  episodes within a single eval, which is a different quantity. For robustness evidence
  use agreement across datasets or splits instead.

## Where to Start Reading

- `scripts/experiments/analysis/_cross/README.md` — index of every analysis folder,
  which findings file is current, and which are superseded.
- The 23 analyses retired on 2026-07-26 are **not** in the working tree: branch
  `archive/retired-analyses-2026-07` and tag `archive/retired-analyses-2026-07-26`.
  Tag `pre-cleanup-2026-07-26` is the pre-consolidation state. Earlier superseded work
  *is* still in the tree under `scripts/experiments/analysis/archive/`, which has its
  own `README.md` mapping both sets.

## Experiment Conventions

- Keep experiments atomized: each experiment should be self-contained, with config, runner, and notes in its own subfolder.
- Producing runs and interpreting them are kept apart, in two name-aligned trees:
  - `scripts/experiments/setup/<name>/` — configs, launch/eval scripts, and the
    `README.md` describing how to reproduce the run. Nothing downstream.
  - `scripts/experiments/analysis/<name>/` — notebooks and plotting/table code,
    findings, plus `data/` and `figures/` subfolders.
- The trees are independent: an experiment with no analysis yet, or an analysis
  with no dedicated experiment folder, is normal. Do not create empty shells.
- Name new folders identically on both sides. One pair is already out of step —
  `setup/nm_ladder_order_robustness-jul_23/` vs `analysis/nm_ladder_order_robustness/`
  — so match by prefix when a lookup misses, and do not add more dated suffixes.
- Findings files (`RESULTS.md`, `FINDINGS.md`) live with the analysis; the
  `README.md` that explains how to run the experiment stays with the setup.
- Eval CSVs under `data/` and figures under `figures/` are committed on purpose —
  they are the evidence behind each findings file. Artifacts elsewhere stay
  ignored, and `.githooks/pre-commit` rejects anything over 25 MB. Enable it once
  per clone (including on Tucker) with `git config core.hooksPath .githooks`.
- Pull results from cluster logs into a notebook rather than leaving loose result files at the repo root.
- Prefer the shared train/eval harness over one-off scripts.
- Task-name aliases accepted by `-task_name` (the full map, `experiments/params.py`):
  `nm`→`neighbor_matching`, `cl`/`same_graph`→`contrastive`,
  `fp`/`mfp`→`masked_feature_prediction`, `slp`→`static_link_prediction`,
  `reg`→`regression`, `mix`→`nm_fp_cl`, `e4`→`e4_multi`. Anything unrecognised is
  passed through **unmapped**, so a typo fails late rather than loudly.
- `lp` and `pl` are *not* CLI aliases — they are analysis-side shorthand only
  (`pl` = classification, `lp` = temporal link prediction, as column labels in
  `analysis/` scripts). On the command line write `temporal_link_prediction` and
  `classification` in full.
- Checkpoints usually live under `state/<run_name>/checkpoint/state_dict_<step>.ckpt`,
  written every `checkpoint_step` plus a terminal one at the true step count.
  Total steps are `epochs × dataset_len_cap`, not an explicit flag.
- Eval logs usually live under `log/eval_<model>_to_<dataset>_<task>_<shots>shot_<timestamp>/`.
- For checkpoint trajectories, evaluate every `state_dict_<step>.ckpt` in numeric step
  order — but see the final-checkpoint trap above before comparing the last rung of a
  pre-2026-07-26 run against a post-fix one.

## Working Style

- Prefer existing scripts and configs over inventing new command patterns.
- Use dry runs when creating large eval sweeps.
- When adding eval helpers, make them overrideable with CLI args or environment variables.
- If a path only exists on Tucker, say that clearly instead of trying to validate it locally.
