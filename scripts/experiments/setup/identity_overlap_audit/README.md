# Cross-dataset identity-overlap audit

This audit separates two questions that must not be conflated:

1. **Exact account-ID overlap** where graph artifacts retain a comparable,
   platform-global Twitter user identifier.
2. **Exact normalized-biography overlap** as a representation-level proxy where
   identifiers were discarded or namespaced. A stricter proxy counts a shared
   biography only when it has at least 20 normalized characters and occurs once
   in each graph.

The script writes aggregate counts only. It never writes account identifiers,
handles, or biography text. `not_measurable` and `incompatible_platform` cells
remain explicit in the pairwise ID table.

## Tucker run

Use the `bio-embeddings-v001` environment because the audit reads the staged
Parquet provenance and existing embedding artifacts:

```bash
tmux new-session -d -s identity-audit \
  'export PATH="/home/mhchu/miniconda3/bin:$PATH"; \
   bash scripts/experiments/setup/identity_overlap_audit/run_identity_overlap_tucker.sh'
```

The default output is a fresh versioned directory under
`state/identity_overlap_audit/`; an existing directory is rejected. The large
DuckDB scratch database stays untracked beside the aggregate outputs.

## Outputs

- `dataset_inventory.csv`: graph sizes, identifier coverage and bio-hash coverage.
- `pairwise_identity_overlap.csv`: exact/partial exact account-ID evidence or an
  explicit reason the pair is not measurable.
- `pairwise_biography_overlap.csv`: exact non-empty and strict unique-long bio
  proxy counts for Twitter graphs.
- `summary.json`: protocol, provenance, source-manifest digests and headline
  maxima. No row-level identifiers or text are serialized.

The analysis copy and manuscript must call biography overlap a proxy, not proof
of shared identity. The Ukraine-Suspended ID array covers only a subset of graph
nodes, so its exact-ID intersections are lower bounds for the full graph. Hong
Kong and TwiBot-20 retain dataset-internal string namespaces whose compatibility
with Twitter snowflakes is not established; those are not compared as account
IDs.

## Completed run

The completed aggregate is documented under
`scripts/experiments/analysis/graph_characterization/dataset_overlap/identity_overlap_audit/`. Tucker `v002` ran at
commit `c44720c`; the first `v001` attempt failed before scanning data because
the repository root was missing from `sys.path` and is preserved unchanged.
