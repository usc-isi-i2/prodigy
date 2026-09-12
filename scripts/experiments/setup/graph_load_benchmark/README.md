# Graph loading benchmark

Bounded CPU diagnostic on the catalog's `merged-all8` graph. No training or
source-graph modifications. Run in its own worktree and tmux session.

`run.py` uses the production `prodigy` environment and repeats eager graph loading,
ordinary sampler preprocessing, and cached CSR restoration three times. It saves
the exact row pointer, column, and signed edge-ID tensors and verifies their
round trip. Cache creation and verification are excluded from recurring startup
timings; creation time is reported separately. The output directory must be new.

```bash
python scripts/experiments/setup/graph_load_benchmark/run.py \
  --graph /dataMeR1/phil/data/merged/graphs/ukr_rus_covid_midterm_all8_retweet_graph.pt \
  --output /dataMeR1/phil/gfm/prodigy-loadbench/state/loadbench-20260912
```

`mmap_probe.py` requires PyTorch supporting `torch.load(mmap=True)` and matching
torch-sparse/PyG binaries. The isolated CPU benchmark environment on Tucker is
`/dataMeR1/phil/conda_envs/prodigy-loadbench` (PyTorch 2.5.1+cpu,
torch-sparse 0.6.18+pt25cpu, PyG 2.6.1). Production remains unchanged.

```bash
/dataMeR1/phil/conda_envs/prodigy-loadbench/bin/python \
  scripts/experiments/setup/graph_load_benchmark/mmap_probe.py \
  --graph /dataMeR1/phil/data/merged/graphs/ukr_rus_covid_midterm_all8_retweet_graph.pt \
  --cache /dataMeR1/phil/gfm/prodigy-loadbench/state/loadbench-20260912/csr.pt \
  --output /dataMeR1/phil/gfm/prodigy-loadbench/state/mmap-20260912.json
```

This compares eager graph+CSR loading, mapped graph+CSR loading, and mapped loading
with the ordinary sampler's CSR sharing step. Each run samples 210 fixed random
centers with two-hop 9/9 fanouts, a 101-node cap, and actual node-feature gathers.
All modes must yield identical sampled nodes, edges, edge IDs, and features under
the new environment. This is not a cross-version training-equivalence check.

Limitations: OS caches are not flushed. Results exclude full dataset construction,
source-pool creation, feature transforms, worker spawning, GPU/model operations,
and the shared-training launcher's full graph-tensor sharing. Mapped results are
not measurements of cold-disk access or full graph traversal. These scripts are
diagnostics, not a production cache implementation with invalidation/locking.

`reader_concurrency.py` runs two complete mapped-reader processes sequentially,
then two concurrently. Supply the same `--graph` and `--cache` arguments as above,
and a new directory for `--output`. It records wall time including interpreter
startup and process cleanup, plus each reader's stage timings and result hash.
This is one paired scheduling trial, not a GPU or training concurrency benchmark.
