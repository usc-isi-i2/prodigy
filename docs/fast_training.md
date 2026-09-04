# Fast PRODIGY training on Tucker

Use `experiments/run_shared_graph.py` for independent source-restricted NM models
that use the same full graph. The supervisor loads the graph and sampling indices
once in shared CPU memory. Spawned trainers own separate weights, optimizers,
RNGs, source schedules, metrics, and checkpoints. Only GPUs **0–3** are allowed.

Anomaly debugging is off by default in the trainer (`--detect_anomaly True`
restores it). For one model using the normal runner, start with 8–16 loader workers
on idle Tucker. For multiple models, use the launcher's total worker budget.

## Status and validation

Implementation worktree: `/Users/philipp/projects/gfm/prodigy-profile`, branch
`codex/ladder-sampling-profile`, pushed to origin. Tucker worktree:
`/dataMeR1/phil/gfm/prodigy-profile`. Revision `677f50c` passed a full-graph
eight-model smoke run on physical GPU 2 on 2026-09-04: all eight ladder rungs
completed 200 steps with finite, distinct terminal checkpoints. Four loader
workers per model used a total budget of 32. Aggregate throughput after warmup
was 57.9 steps/s across the combined measurement window; shared graph setup
took 144 seconds. This validates concurrent execution, not optimal concurrency,
long-run stability, or an eightfold speedup. See the
[validation report](../scripts/experiments/analysis/evaluation/performance/ladder_sampling_profile/SHARED_TRAINING_VALIDATION.md).

The shared-training, source-subset, and sequential-source-schedule suites passed
22 tests on Tucker. Shared-training checks live in
`experiments/tests/test_shared_graph_training.py`; run them with the `prodigy`
Python using `-m unittest discover -s experiments/tests -p test_shared_graph_training.py`.
They cover spawned shared storage, unchanged source-pool episode draws,
source boundaries, cross-source edge rejection, config/worker-budget checks, and
GPU visibility being assigned before spawning an interpreter.

## Plan before launch

Use a dedicated Tucker worktree. Inspect `tmux ls`, GPU processes, host RAM, and
`/dev/shm` capacity first. Do not pull code into a worktree with running jobs.
Activate `prodigy` and use a fresh output directory:

```bash
export PATH="/home/mhchu/miniconda3/bin:$PATH"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate prodigy
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export WANDB_MODE=offline
python experiments/run_shared_graph.py \
  --configs scripts/experiments/setup/nm_ladder_nhop2/configs/train_ordA_r*.yaml \
  --gpus 2 --models-per-gpu 8 --worker-budget 32 \
  --run-dir /dataMeR1/phil/gfm/prodigy-profile/log/ladder_A_shared_YYYYMMDD \
  --dry-run
```

This prints the resolved configs and their unchanged training budgets without
loading data or creating output directories. Remove `--dry-run` to execute only
after validation and authorization. Long training runs belong in tmux; include the
PATH export and environment activation **inside** its command. The user normally
launches long sweeps; this guide does not itself authorize a launch.

For a bounded validation, add `--smoke-steps 200` and use a separate
output directory. Smoke mode labels runs `smoke_`, overrides the budget, and
disables evaluation. Its checkpoints are not completed ladder results. It rejects
blocked-source schedules to avoid silently truncating source exposure.

## Concurrency controls

- `--gpus 2 --models-per-gpu 8`: eight independent models on one GPU.
- `--gpus 2 3 --models-per-gpu 4`: eight models across two owned GPUs.
- The default is two models per GPU, not a measured optimum.
- `--worker-budget 32` bounds concurrent training-loader workers. Eight models get
  four each; one model gets at most sixteen by default. Config worker values are
  overridden and the effective values are saved.
- `--workers-per-model N` explicitly overrides that allocation, but must fit the
  budget. Zero is allowed; exact training-state resume requires zero workers.
- `--threads-per-model 4` controls tensor threads in trainers. CPU loader workers
  use PyTorch's single-thread default. Do not multiply sixteen workers by every
  model without checking CPU capacity and measured throughput.
- Queued jobs reuse the resident graph when slots become free.
- Trainers and loaders use `spawn`; the supervisor never initializes CUDA.
  Start from an unmasked environment (unset `CUDA_VISIBLE_DEVICES`) and select
  physical GPU IDs with `--gpus`. Each trainer is then isolated to its assigned
  GPU and sees it as logical `cuda:0`; outputs record the physical GPU too.
- GPU visibility is assigned before each interpreter starts. A concurrent CUDA
  context preflight runs before graph loading. Add `--preflight-only` to check
  the selected slots without loading the graph or training.

The single-model H100 profile measured 0.45 GiB of peak tensor allocations and
14.6 ms per synchronized GPU step with anomaly debugging off. Loader throughput
was 12.2 episodes/s with two workers, 46.9 with eight, and 60.1 with sixteen.
Those isolated measurements establish neither eight-model throughput nor exact
aggregate GPU memory. CUDA context/library overhead is additional. See
[profiling findings](../scripts/experiments/analysis/evaluation/performance/ladder_sampling_profile/FINDINGS.md).

## Supported protocol and outputs

The shared launcher supports `dataset: covid19_twitter` graph-format NM,
`original_features: true`, `structural_features: none`, and
`neighbor_sampling_episode_source: graph_id`. It checks that background edges
cannot cross source IDs. Per-model source subsets and blocked schedules may differ.
Graph paths, edge/feature views, hop/fanout settings, split settings, and unknown
new options must match across configs or the group is rejected. Compatible NM
holdout CSR indices are shared too. Label downsampling and target-feature
transforms are currently unsupported.

The source artifact is unchanged. Human-readable user IDs are omitted from the
in-memory training wrapper to avoid repeatedly serializing millions of strings;
numeric node IDs, features, and topology remain. Shared source pools retain the
same ordered node IDs. Different worker counts can change stochastic training
samples and do not promise bitwise historical replay.

Common experiment overrides go after `--`, e.g. `-- --detect_anomaly True`.
Outside explicit smoke mode/overrides, shots, queries, fanouts, training steps,
split views, and checkpoint schedules are preserved.

The run directory holds `manifest.json`, per-model `job_NNN/effective_config.json`,
`console.log`, and `result.json`, plus normal `state/<unique-run>/checkpoint/` and
`log/<unique-run>/` outputs. Results include shared-storage checks, peak allocated
GPU memory, checkpoint paths, and steady step throughput after ten warmup steps.
That timing includes normal loop logging and intervening checkpoint/eval pauses,
but excludes initial startup and terminal checkpoint saving.

`status.json` records final completion or failure/interruption. A failed trainer
stops the remaining trainer process groups. Existing output directories are never
overwritten. Use absolute checkpoint paths across worktrees. Existing resume
checks still reject exact training-state resume with more than zero workers.
