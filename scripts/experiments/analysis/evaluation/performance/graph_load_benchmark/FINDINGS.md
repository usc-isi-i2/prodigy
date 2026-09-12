# Graph startup benchmark — 2026-09-11

## Scope

CPU-only Tucker trial, branch `codex/graph-load-benchmark`, worktree
`/dataMeR1/phil/gfm/prodigy-loadbench`. Graph: `merged-all8`, 34,482,222 nodes,
191,523,118 directed edges, 768 node features. Source artifact is
111,386,034,033 bytes, including a 1,840,561,014-byte pickle metadata entry.
The source graph was not modified. No GPU jobs were launched.

## CSR caching in production PyTorch 2.0.1

Four CPU tensor threads. The first baseline took 152.88 s (135.26 s loading,
15.73 s preprocessing, 1.88 s sharing). OS caches were not flushed, so do not
compare that initial run directly against later cache hits as a causal speedup.

Matched later repeats:

| Stage | Baseline mean, seconds | Cached mean, seconds |
|---|---:|---:|
| Graph load | 73.07 | 73.07 |
| CSR build / reload | 16.58 | 2.48 |
| CSR shared-memory preparation | 2.01 | 1.93 |
| Total | 91.66 | 77.48 |

**CSR caching alone gives 1.18x startup speedup, saving about 14 seconds (15%).**
The cache is 6,404,598,691 bytes (~5.96 GiB); writing it took 6.92 seconds,
excluding the adjacency build. Full row-pointer, column, and signed edge-ID
tensors compared exactly after serialization. All six raw observations are in
[`data/csr_timings.json`](data/csr_timings.json).

This measures graph load plus sampler index preparation, not all dataset setup
or training startup. It refutes the earlier speculative 2–5x estimate for CSR
caching alone on this graph.

## Memory mapping in isolated PyTorch 2.5.1+cpu

Two repeats per mode with reversed mode order on the second pass. Each run loads
the same graph and saved CSR, then actually samples 210 fixed random centers with
two-hop 9/9 fanouts and reads their node features. All six runs produced the same
SHA-256 over sampled nodes, edges, signed-ID-derived edge IDs, and feature values.

| Mode | Mean seconds through sampled feature access | RSS after access |
|---|---:|---:|
| Eager graph + cached CSR + CSR sharing | 78.69 | 115.20 GiB |
| Mapped graph + mapped CSR | 28.28 | 7.81 GiB |
| Mapped graph + mapped CSR + CSR sharing | 29.94 | 13.58 GiB |

Mapping with the existing CSR-sharing step gives **2.63x** speedup against eager
cached loading in the same newer environment. Relative to the production warm
baseline including CSR construction (91.66 s), the combined approach is about
**3.06x** faster. That cross-environment comparison has slightly different scope:
the newer probe additionally includes sampled feature access. The old/new eager
cache-hit controls are similar (77.48 vs 78.69 seconds).

The remaining ~28 seconds are in graph deserialization despite tensor mapping;
the archive retains 1.84 GB of pickle metadata. Splitting unused metadata from
runtime tensor storage is a plausible next optimization, not measured here.

These are warm-filesystem-cache, CPU graph-access measurements, **not** a complete
training-startup or throughput benchmark. They omit feature transformations,
source-pool creation, workers, GPU/model operations, and the shared-training
launcher's full graph-tensor sharing. The sampled centers are a fixed synthetic
workload, not a source-balanced NM episode. RSS is resident process memory at the
measurement point, not peak or unique memory across processes. No production
environment was upgraded, and full model compatibility with the newer environment
was not tested. See [`data/mmap_timings.json`](data/mmap_timings.json).

## Scheduling and reuse

Two independent mapped-reader processes, each sampling the same 210-center
workload, completed in **63.72 s sequentially versus 31.87 s concurrently**:
**2.00x throughput** in this single paired trial. Wall times include interpreter
startup and process cleanup. Each reader's timed work stayed around 28.3–28.4 s;
there was no material per-reader contention at concurrency two. All four output
hashes matched each other and the earlier single-reader comparison. Each reader
held ~7.82 GiB RSS after access; summing RSS does not measure unique physical RAM
because mapped pages can be shared. This trial used mapped CSR without an extra
CSR shared-memory copy, warm OS caches, four tensor threads per reader, and no
GPU work. It does not establish an optimum beyond two readers or cold-cache
behavior. See [`data/reader_concurrency.json`](data/reader_concurrency.json).

Existing scheduling paths:

- `experiments/run_shared_graph.py` loads a compatible NM dataset once and reuses
  it for both concurrent and queued trainers. Group compatible graph/view/split
  configurations, bound the total loader worker budget, and use owned GPUs 0–3.
- `scripts/eval/pair_link_sweep.py` already loads one graph/adjacency before its
  checkpoint/model loop. Use it for graph-compatible static-LP sweeps.
- `scripts/eval/eval_ckpts_all_graph_tasks_tucker.py` launches independent
  processes, each with its own graph loading. More slots are not resident reuse.
  Its FIFO completion wait can also leave a later completed slot idle. A future
  scheduler improvement is free-slot dispatch plus resident per-graph workers.
- RQ1 compact topology caches still reconstruct features from a loaded full
  graph. Its separate-process adaptation cells also pay repeated startup costs.

For eight serial independent loads, avoiding seven repeated 91.66-second startup
stages would save roughly 10.7 minutes **arithmetically**, before accounting for
shared-launcher overhead. This is not a measured eight-model scheduling speedup.

## Recommendation

Use mapped tensors plus a versioned CSR cache for repeated CPU graph access, and
retain graph residency across compatible queued jobs wherever possible. Two
independent mapped readers worked well in this bounded trial. Production work
still requires cache identity/invalidation and concurrency-safe creation, actual
dataset integration, worker/shared-memory checks, and GPU-environment validation.
Do not multiply the ~3x startup reduction and ~2x two-reader throughput result
into a claim of 6x faster training.

Artifacts and environments remain on Tucker under the paths in the setup README;
the generated CSR is inside `state/loadbench-20260912/csr.pt`. The production
`prodigy` environment and the original graph remain unchanged. Local analysis and
scripts are in `/Users/philipp/projects/gfm/prodigy-loadbench` on the same branch.

See [reproduction instructions](../../../../setup/graph_load_benchmark/README.md).
