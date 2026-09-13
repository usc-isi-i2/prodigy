---
name: prodigy-graph-catalog
description: Add, remove, move, rename, verify, or document PRODIGY graph datasets and graph artifacts, identifiers, metadata, provenance, or supported tasks. Use whenever docs/graph_catalog.json or a complete graph registry is involved.
---

# PRODIGY Graph Catalog

Treat `docs/graph_catalog.json` as the single source of truth for graph names,
artifact paths, source composition, provenance, statistics, and evaluation
capabilities.

## Workflow

1. Read the target catalog entry and the code paths that consume it.
2. Update the catalog first when adding, removing, moving, or changing a graph.
3. Use `canonical_name` in prose and new documentation. Use `dataset_key` for
   compatibility with existing configs, CLI arguments, loaders, logs, and historical
   artifacts; do not rename those artifacts retroactively.
4. Resolve every `relative_path` beneath the catalog's `data_root`. The only current
   Tucker data root is `/dataMeR1/phil/data`; do not introduce another root.
5. Keep byte size, node/edge counts, features, labels, tasks, source locations,
   metadata sidecars, construction provenance, and status current.
6. Verify artifact facts read-only on Tucker. Use the `prodigy-tucker` skill for
   cluster work.
7. Advance the catalog's top-level `last_verified` only for facts actually checked.
   There is no per-entry verification date. Represent unknown facts with explicit
   `null` values or notes instead of guesses.
8. Make code that needs the complete graph inventory read the catalog. Hard-coded
   experiment-specific subsets are allowed.

Also inspect `docs/graph_reference.md` and any graph-specific provenance document
linked by the entry when they are relevant to the requested change.
