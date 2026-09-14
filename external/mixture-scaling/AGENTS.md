# Agent instructions

- This repository owns the standalone GraphSAGE mixture-scaling experiments.
- Graph artifacts remain under `/dataMeR1/phil/data`; do not copy or overwrite them.
- State, logs, and raw result shards remain under
  `/dataMeR1/phil/gfm/mixture-scaling/{state,log,results}`.
- Never delete or replace an existing run directory. Launchers must skip complete runs
  and refuse ambiguous partial runs.
- Use Tucker GPUs 0, 1, 2 and 3 for this project. Leave GPUs 4–7 untouched.
- Prefer a shared work queue across available owned GPUs. Check process ownership
  before scheduling; do not interrupt unrelated jobs. Benchmark throughput, not only
  instantaneous utilization, and preserve sampling and optimization semantics.
- Use the Tucker `prodigy` conda environment for GraphSAGE training and evaluation.
- Source changes move through git. Do not hand-copy source files to Tucker.
- The primary protocol uses source-confined random-walk/edge pairs, uniform source
  rotation, seed 0, fixed ten-labels-per-class probes, and checkpoints 100/300/900/2500.
- Select convergence using SSL validation only. Never select a checkpoint on downstream
  test performance.

