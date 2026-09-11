# MLP endpoint prefetch benchmark

2026-09-11, Tucker GPU 1, implementation `478bafd`. Real Ukraine/COVID features
and canonical LP split caches; Ukraine resident on GPU and COVID gathered from
CPU, matching the two-source memory constraint. Other ladder workers remained
active on GPUs 2 and 3. Each mode used the same initialization, source rotation,
batch/negative sampling, FP32 model, AdamW and gradient clipping. Timing covers
500 updates after 20 warmup updates and excludes graph loading. The benchmark
asserted exact final weight equality for both prefetch settings.

| In-flight batches | Updates/sec | Timed seconds | Max weight difference |
|---:|---:|---:|---:|
| 0 (synchronous) | 99.44 | 5.028 | 0 |
| 8 | 269.25 | 1.857 | 0 |
| 16 | 243.90 | 2.050 | 0 |

Eight is the selected default: 2.71× synchronous throughput in this short,
two-source benchmark. This is not an end-to-end runtime claim or a guarantee of
full GPU utilization. Loading, validation, W&B startup, source count and concurrent
jobs also affect wall time. Raw measurements: `data/prefetch_benchmark.json`.
