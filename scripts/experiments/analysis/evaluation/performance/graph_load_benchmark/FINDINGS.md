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

## Scheduling and reuse

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

See [reproduction instructions](../../../../setup/graph_load_benchmark/README.md).
