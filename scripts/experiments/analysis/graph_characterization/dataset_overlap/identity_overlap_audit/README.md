# Identity-overlap audit analysis

This folder contains only aggregate, privacy-preserving outputs from the Tucker
run documented in `../../setup/identity_overlap_audit/`. No account identifiers,
handles, biography strings, or graph-level DuckDB tables are committed.

- `data/dataset_inventory.csv`: identifier and biography coverage.
- `data/pairwise_identity_overlap.csv`: all 36 unordered graph pairs, retaining
  explicit unmeasurable and cross-platform cells.
- `data/pairwise_biography_overlap.csv`: 28 unordered Twitter-graph pairs.
- `data/summary.json`: source/output hashes, temporal ranges, and protocol.
- `FINDINGS.md`: interpretation and manuscript boundary.

Canonical run: `/dataMeR1/phil/gfm/prodigy-identityaudit/state/identity_overlap_audit/v002`
at Prodigy commit `c44720c`. Failed pre-scan `v001` is preserved; it emitted no
result rows because direct execution lacked the repository import path.
