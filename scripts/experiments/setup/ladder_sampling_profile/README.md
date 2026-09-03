# Ladder sampling profile

Profiles the full all-eight graph with the production two-hop NM ladder config
(30-way, 3-shot, 4-query, 9/9 fanouts, 101-node limit). This is a bounded diagnostic,
not a ladder arm. No model checkpoint or benchmark accuracy is produced.

Run in its own Tucker worktree on `codex/ladder-sampling-profile`. Only GPU 2 or 3
is accepted. Check free RAM, GPU processes, and tmux sessions before launch.

```bash
export PATH="/home/mhchu/miniconda3/bin:$PATH"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate prodigy
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export WANDB_MODE=offline
python scripts/experiments/setup/ladder_sampling_profile/profile_pipeline.py \
  --output /dataMeR1/phil/gfm/prodigy-profile/log/profile_20260903 \
  --device 2
```

Use a tmux session for the full graph load. The output directory must not exist.
All counts, config path, CPU thread counts, and loader worker counts are overrideable.
`--cpu-only` omits model initialization and GPU measurements.

Measurements:

- Full graph load plus adjacency preprocessing, and separate loader setup.
- Per-source CPU stages with 4 and 1 tensor threads. Each source has 4 warmup
  episodes, one cProfile episode excluded from stage statistics, and 7 measured
  episodes by default. Existing sampling calls are timed without changing draws.
  `get_subgraph_seconds` includes `neighborhood_seconds`; do not sum them.
- Loader-only throughput with 0, 2, 4, 8 workers; each test has four warmup and 32
  timed episodes. This tests producer capacity, not GPU-overlapped throughput.
  CPU worker tests finish before CUDA initialization.
- Production model, loss, and optimizer on GPU, using saved per-source CPU batches.
  Timing boundaries synchronize CUDA; transfer, forward/loss, backward, and
  optimizer are separate. CPU clones to protect saved batches are outside timing.
  Compare anomaly detection on (the production runner default) and off; optimizer
  steps are disposable. No logging/checkpoint overhead is included in these times.

`metadata.json` records revision, effective config, hardware, and arguments;
`measurements.json` is saved after every row; per-source `.txt` files hold cProfile
call stacks. The full graph and adjacency are loaded once. All source data remain
unchanged. Timings from a short profile are diagnostics, not confidence intervals
for experimental accuracy or evidence that multi-model throughput scales linearly.
