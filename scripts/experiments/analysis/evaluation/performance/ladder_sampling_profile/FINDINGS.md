# PRODIGY ladder pipeline profile — 2026-09-03

The two-worker loader supplies substantially fewer episodes per second than the
isolated GPU step can consume. Most CPU preparation time is spent constructing
210 separate subgraphs and combining them. Serial center selection is a small
fraction of this cost. Increasing loader workers helps; global anomaly debugging
also adds measurable overhead on both CPU and GPU.

## Measured results

Tucker was idle before the run. Hardware: H100 80 GB (physical GPU 2), 384 logical
CPU cores, approximately 1.4 TiB available host RAM. PyTorch 2.0.1+cu118.
The existing all-eight graph has 34,482,222 nodes, 191,523,118 directed edges,
and 768-dimensional features. Loading plus CSR preprocessing took 127 seconds
in the first run and 97 seconds in the follow-up; this difference is not an
anomaly-debugging speedup claim. The OS file cache was not flushed.
Loader setup took 2.7 seconds. The graph was loaded once per profiling process.

The production two-hop ladder config has 30 labels × (3 support + 4 query) = 210
subgraphs per episode, 9/9 fanouts, a 101-node limit, batch size one, and balanced
source-confined episodes. All eight sources were profiled separately.

| Measurement | Debugging on (production default) | Debugging off |
|---|---:|---:|
| Typical serial CPU preparation, median of isolated episodes | 169 ms | 142 ms |
| Loader throughput, 0 workers | 5.99 episodes/s | 6.94 episodes/s |
| Loader throughput, 2 workers | 10.40 episodes/s | 12.17 episodes/s |
| Loader throughput, 4 workers | 19.99 episodes/s | 24.14 episodes/s |
| Loader throughput, 8 workers | 36.87 episodes/s | 46.88 episodes/s |
| Loader throughput, 16 workers | not measured | 60.06 episodes/s |
| Synchronized GPU step including transfer, mean | 27.00 ms | 14.64 ms |
| Peak allocated GPU tensor memory | 0.45 GiB | 0.45 GiB |

The loader-only measurements consume batches immediately and exclude worker
startup. Debugging-on uses 32 measured episodes; debugging-off uses 128. Both
have four warmup episodes. These are short producer-throughput measurements,
not full training throughput or a concurrent-model benchmark.

The model has 1,640,514 parameters. GPU tensor allocation excludes CUDA context
and other process/library overhead. This is strong evidence that eight such
models are plausible on an 80 GB GPU by memory capacity; it does not establish
eight-model speedup or exact aggregate memory use.

## Where the CPU time goes

Mean times over 56 isolated episodes with four tensor threads, production
anomaly debugging enabled:

| Stage | Mean milliseconds per episode |
|---|---:|
| Choose centers and positive members, including rejected attempts | 9.1 |
| Neighborhood extraction and edge deduplication | 105.9 |
| Feature gathering and graph object construction outside neighborhood extraction | 21.9 |
| Add pooling supernodes | 20.0 |
| Dataset traversal and timing-wrapper remainder | 2.0 |
| Collation and metagraph/sequence assembly | 26.2 |
| Total | 185.1 |

The mean exceeds the 169 ms median because occasional pauses affect individual
episodes. These stage measurements include lightweight timing wrappers;
cProfile episodes themselves are excluded from the timing distributions.
`get_subgraph_seconds` includes `neighborhood_seconds`, so those two raw fields
must not be added together.

Center selection costs approximately 1.9 ms on election2020-political and
17.7 ms on hongkong. Average rejected-center counts range from 0.4 to 332 per
episode. Even where many candidates are rejected, this is not the dominant
CPU stage. Four versus one tensor threads provides no consistent large benefit;
this profile does not justify changing the thread count.

The call profiles identify a concrete debugging cost. The production runner
sets `torch.autograd.set_detect_anomaly(True)` at import. During CPU subgraph
construction, edge deduplication's `torch_scatter.segment_min_csr` path records
Python stack traces. For the profiled ukraine episode, disabling debugging
removes `traceback.format_stack` from the top calls, reduces total Python calls
from 284,557 to 126,189, and reduces profiled cumulative edge deduplication time
from about 90 to 34 ms. Those cProfile times demonstrate the mechanism; they are
not substituted for the unprofiled timings above.

With debugging off, work remains spread across neighborhood sampling,
deduplication, tensor concatenations, feature gathering, graph object creation,
and batch collation. Batched sampling/assembly is a reasonable later target,
but adding workers already improves producer throughput substantially.

## GPU measurements and implications

Means over 16 measured disposable optimizer steps, covering each source twice,
with four warmup steps. All stage boundaries explicitly synchronize CUDA.

| GPU pipeline stage | Debugging on | Debugging off |
|---|---:|---:|
| CPU-to-GPU transfer | 5.8 ms | 6.0 ms |
| Forward and loss | 10.2 ms | 3.6 ms |
| Backward | 10.4 ms | 4.5 ms |
| Optimizer | 0.6 ms | 0.6 ms |

These are synchronized wall times including launch overhead, not pure GPU kernel
durations. Batches are cloned on CPU outside timing because PyG `.to()` mutates
its object; every timed transfer starts with a CPU batch. GPU peak allocation
is measured after warmup and includes optimizer state. The profiler uses the
production trainer/model/loss but omits full train-loop logging, checkpointing,
and validation. No benchmark-result checkpoint or accuracy was produced.

Recommended order of work:

1. Make anomaly debugging opt-in for normal training; retain it for debugging.
2. Try 8–16 loader workers for a single run, then allocate a bounded shared CPU
   worker budget when multiple models run together.
3. Add one shared, read-only full graph and adjacency store for independent
   source-restricted trainers. Share graph tensors, not weights or optimizer state.
4. Measure actual 1/2/4/8-model throughput; memory feasibility alone does not
   determine the best concurrency. Prefer fewer total CPU workers if contention
   appears, rather than blindly multiplying 16 workers by eight models.

## Reproduction and provenance

Setup: `scripts/experiments/setup/ladder_sampling_profile/README.md`.
Both runs used separate `ladder-profile` tmux sessions in
`/dataMeR1/phil/gfm/prodigy-profile`, branch `codex/ladder-sampling-profile`.
Baseline profiler revision: `2f3ebed1b59b0977b0d53b161608fdae496a5177`.
The follow-up used `5ef0e0ed` with `--cpu-only --disable-cpu-anomaly
--loader-episodes 128 --workers 0,2,4,8,16`.

Tucker outputs:

- `/dataMeR1/phil/gfm/prodigy-profile/log/profile_20260903/`
- `/dataMeR1/phil/gfm/prodigy-profile/log/profile_20260903_noanomaly/`

Local evidence: `data/` holds baseline measurements, effective parameters,
hardware/revision metadata, and per-source call profiles. `data/noanomaly/`
holds the follow-up. Each run has 112 CPU-stage observations: seven measured
episodes × eight sources × two thread counts. Raw names use the merge artifact's
source identifiers for compatibility.

To regenerate summary tables with standard-library Python:

```bash
python scripts/experiments/analysis/evaluation/performance/ladder_sampling_profile/summarize.py
python scripts/experiments/analysis/evaluation/performance/ladder_sampling_profile/summarize.py \
  --input scripts/experiments/analysis/evaluation/performance/ladder_sampling_profile/data/noanomaly/measurements.json
```

The local worktree is `/Users/philipp/projects/gfm/prodigy-profile`. Production
training implementation was not modified by this profiling work.
