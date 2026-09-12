# Nonzero mini structural sampling pilot

Compare existing uniform-node and edge-endpoint 500k minis against uniformly
restarted walks of lengths 1, 2, 4, 8. All candidate topologies are the complete
induced subgraph of the selected nodes. Walks use distinct undirected non-self
neighbors. Roots are uniform over all nonzero nodes, including isolates; isolate
walks stay at their roots. The exact cap keeps the first 500k distinct nodes in
walk-major traversal order. At most the final walk is cut short.

No candidate feature files, model training, or new canonical graph artifacts are
created. Selection files contain row IDs in both parent and original graphs.
The parent is loaded once with mmap, adjacency is built once, and degree/component
statistics are shared across candidates. Two graphs may run concurrently with
four tensor threads each in isolated Tucker tmux sessions.

Screening uses seed 0. The best two walk lengths per graph are repeated with seed
1, ranked by a disclosed diagnostic score: absolute log density ratio + degree
KS + absolute isolate-fraction error + absolute largest-component-fraction error.
Also report degree quantiles, log-degree Wasserstein distance, parent degrees of
selected nodes (selection bias), component counts and node-weighted component
size bins. Do not interpret this heuristic score as proof of unbiased sampling.

Use `/dataMeR1/phil/conda_envs/prodigy-loadbench/bin/python`:

```bash
python -m unittest discover -s scripts/experiments/setup/nonzero_mini_sampling_pilot -p test_run.py
python scripts/experiments/setup/nonzero_mini_sampling_pilot/run.py \
  --dataset ukr_rus_twitter \
  --output /dataMeR1/phil/gfm/prodigy-walk-mini-pilot/state/ukraine
python scripts/experiments/setup/nonzero_mini_sampling_pilot/run.py \
  --dataset covid19_twitter \
  --output /dataMeR1/phil/gfm/prodigy-walk-mini-pilot/state/covid
```

`results.json` is incremental; only `completed: true` denotes a finished pilot.
