# Eight-model shared-graph validation — 2026-09-04

Eight independent two-hop NM ladder models completed 200 steps each on Tucker
physical GPU 2 using one full graph in shared CPU memory. Revision: `677f50c`
on `codex/ladder-sampling-profile`. Each model used its original order-A rung's
source subset, independent parameters and optimizer, and four loader workers
(32 total). Smoke mode shortened the training budget and disabled evaluation.

## Measurements

| Measurement | Result |
|---|---:|
| Completed model steps | 8 × 200 = 1,600 |
| Shared graph loading, preprocessing, and sharing | 144.2 s |
| Aggregate throughput after each model's ten warmup steps | 57.9 steps/s |
| Aggregate measurement window | 1,520 steps / 26.24 s |
| Overlap of all eight models' measured intervals | 15.05 s |
| Per-model throughput within its own measured interval | 8.77–9.79 steps/s |
| Training phase including startup and terminal saving | 36.12 s |
| Largest per-model peak allocated GPU tensor memory | 0.475 GiB |

The aggregate denominator spans the earliest model's measured start through the
latest model's measured finish. Summing the eight individual rates yields 74.3
steps/s, but their windows differ; use the conservative 57.9 combined-window rate
when quoting this run. Graph setup is excluded from both rates. The training
phase starts at the earliest trainer's recorded `training_started` and ends at
the latest trainer's `completed` timestamp.

During simultaneous training, a live GPU snapshot showed 15,174 MiB used on GPU 2
and 65% utilization; `/dev/shm` showed about 111 GiB used. These are observations,
not sampled peaks. The downloaded `gpu_snapshot.txt` was taken later during
teardown and shows lower memory use. GPU context/library overhead is additional
to PyTorch's tensor-allocation metric. GPUs other than 2 remained unused by this
run.

This short test establishes that eight models can train together with the full
shared graph. It does not establish the best 1/2/4/8-model concurrency, an
eightfold speedup, production accuracy, or stability over a full training run.

## Correctness and startup checks

- Every trainer confirmed that graph tensors and sampling indices were shared.
- The full-graph edge check rejected any cross-source edges before training;
  this artifact passed. Source-specific pools constrain episode centers, and
  disconnected source components constrain sampled neighborhoods.
- All eight terminal model checkpoints contained finite tensors and distinct
  SHA-256 hashes; corresponding training-state metadata recorded 200 completed
  steps. Exact resume is not supported with these nonzero worker counts.
- The shared-training, source-subset, and sequential-source-schedule suites
  passed 22 tests on Tucker, including actual spawned shared storage and sampled
  source-boundary checks.

The first startup attempt failed with CUDA device availability errors. The fix
assigns `CUDA_VISIBLE_DEVICES` before spawning each interpreter, rather than
inside its target after imports/unpickling. A concurrent tiny CUDA-allocation
preflight now runs before loading the large graph. The successful run used that
fix and passed the eight-context preflight. Start the launcher without an
inherited GPU mask and select physical devices with `--gpus`.

## Reproduction and evidence

Use [the training guide](../../../../../../docs/fast_training.md) for environment
activation, authorization, worktree, and tmux requirements. The tested command,
from `/dataMeR1/phil/gfm/prodigy-profile`, was:

```bash
python -u experiments/run_shared_graph.py \
  --configs scripts/experiments/setup/nm_ladder_nhop2/configs/train_ordA_r*.yaml \
  --gpus 2 --models-per-gpu 8 --worker-budget 32 \
  --run-dir /dataMeR1/phil/gfm/prodigy-profile/log/shared8_smoke_20260904_v2 \
  --smoke-steps 200
```

Choose a fresh output directory for another run. Checkpoints and complete logs
remain in that Tucker directory. The failed first attempt remains in the sibling
`shared8_smoke_20260904` directory. Neither is a completed ladder experiment.

Small result artifacts are committed under `data/shared8_smoke/`: manifest,
effective configs, per-model results, completion status, CPU-only checkpoint
checks, and the generated summary. Recompute and check the summary with:

```bash
python scripts/experiments/analysis/evaluation/performance/ladder_sampling_profile/summarize_shared_smoke.py \
  --input scripts/experiments/analysis/evaluation/performance/ladder_sampling_profile/data/shared8_smoke
```

Local worktree: `/Users/philipp/projects/gfm/prodigy-profile`.
