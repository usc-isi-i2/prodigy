# Fast ladder worker profile — 2026-09-11

Disk I/O was not the bottleneck in the observed steady-state window. Both new workers (GPU 0/rung 9 and GPU 1/rung 8) read 0 kB/s, had zero major page faults and zero reported I/O delay. They consumed approximately 4.2 CPU cores each, including substantial system time, with roughly 570–590k minor faults/s. Minor faults are not disk reads; these observations are consistent with substantial memory allocation/page-management activity.

A short real-data rung-9 replay measured 180 updates after 36 warmup updates, using the production prefetch path and CPU feature representation. The live GPU-0 worker was briefly paused for measurement and resumed successfully; subsequent logs advanced to step 194000. Other GPUs remained active.

| Host-side phase | Share of measured wall time |
| --- | ---: |
| Next batch, including sampling/staging/waits | 37.4% |
| Forward, backward, gradient clipping, optimizer calls | 51.9% |
| Finite-loss check and scalar logging read | 8.1% |
| Zero-grad and remaining overhead | 2.6% |

These are host-call timings, including CUDA dispatch and waits, **not pure GPU kernel times**. Validation and W&B were excluded. The replay is short and its throughput should not be used as the live training rate. CPU COVID feature preparation dominated preparation-thread time, but overlaps the main thread and cannot be added to the percentages above.

Batch staging remains a significant cost, alongside model-call and synchronization overhead. Reusing batch buffers/reducing allocation churn and reducing per-step dispatch/synchronization are candidates for the next measured optimization; faster storage is not supported as a remedy by this sample.

Raw timings: `data/profile_r9.json`. Probe: `src/mixture_scaling/profile_fast_ladder.py`. Branch: `codex/mlp-profile`; local worktree: `/Users/philipp/projects/gfm/mixture-scaling-mlp-profile`; Tucker worktree: `/dataMeR1/phil/gfm/mixture-scaling-mlp-profile`. Active training source was not modified by the probe.
