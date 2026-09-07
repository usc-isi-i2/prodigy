# Role x topology follow-up

Bounded follow-up to the September 6 paper review. No new training, source
selection or checkpoint search. Reuse all six existing HK/UKR original-sort
controls (three initialization seeds) at step 2500. Evaluate covid-political,
election2020 and twibot20, both cached episode streams, all 128 episodes.

The full design is query topology x support topology, each intact / removed /
rewired. Each rewiring draw is shared across models and roles. Three draws give
19 unique conditions per model/target/stream (four deterministic cells plus
five rewiring-involving cells per draw), or 684 cells total.

The null uses finite double-edge swaps within each sampled background subgraph.
Every node's in- and out-degree is preserved. If a subgraph is fully reciprocal,
swap reciprocal pairs together. Preserve existing self loops; reject parallel
edges. The cached graphs have nonconstant edge attributes, but this SAGE model
ignores them: verify every encoder layer's attribute projection is absent and
check first-batch attribute-zeroing parity on every model/target/stream. Keep
the original attribute slots; this is not a weighted-edge rewiring experiment.
The generic helper rejects nonconstant attributes unless explicitly authorized
by that verified attribute-blind contract. Node identities, features, pooling edges,
labels and weights do not change. Request five successful swaps per movable
edge with at most 20 attempts per requested swap. This is not uniform sampling,
does not preserve connectedness, and is not an isolated homophily manipulation.
Record accepted swaps and final edge turnover, including zero-change cases.

Primary contrasts: support removal conditional on intact versus removed query
edges; query removal conditional on intact versus removed support edges; and
the same contrasts replacing removal with rewiring. Report every target, seed
and stream. AUC nonadditivity is performance-scale interaction, not proof that
the logits are nonadditive. Retain accuracy, F1, NLL and per-query logits.

Record shared input degrees and edge-feature similarity, support within-class
dispersion, between-class cosine, representation movement, and learned label
vector separation/movement. These are descriptors, not demonstrated mediators.
The topology transformations never access labels. Representation descriptors
use only support labels. No query outcomes choose randomizations or conditions.

## Run

Use an isolated Tucker worktree synced through Git. The runner refuses an
existing output directory and uses four CPU threads with GPUs hidden.

```bash
bash scripts/experiments/setup/target_performance_mechanisms/run_role_topology_tucker.sh --output log/role_topology_full_20260907 --dry-run
bash scripts/experiments/setup/target_performance_mechanisms/run_role_topology_tucker.sh --output log/role_topology_smoke_20260907 --targets covid_political --seeds 0 --max-batches 2 --rewire-draws 1
tmux new-session -d -s mechanism-role-topology 'cd /dataMeR1/phil/gfm/prodigy-role-topology && export PATH="/home/mhchu/miniconda3/bin:$PATH" && bash scripts/experiments/setup/target_performance_mechanisms/run_role_topology_tucker.sh --output log/role_topology_full_20260907 > log/role_topology_full_20260907.log 2>&1'
```

Run dry-run and smoke before the full command; ensure `log/` exists. Smoke uses
both streams but is explicitly not a full-panel result. Full-run baselines must
match saved metrics to 1e-6. First-batch direct role-specific rewiring forwards
must match factored outputs; untouched-role encodings and all input/weight
hashes are checked. Full numeric outputs are written after each target/stream;
`DONE.json` is written only after the requested grid completes.

`test_role_topology` covers directed and reciprocal degree preservation,
unrewirable/empty graphs, no new loops/duplicates, role scope, label blindness,
all nine direct-forward comparisons, and nested deletion doses. The dose helper
is tested but dose-response is not part of this initial 684-cell campaign.

This follows known target results, not an untouched-domain confirmation. The
three rewiring draws and the two episode streams are not new training seeds.
