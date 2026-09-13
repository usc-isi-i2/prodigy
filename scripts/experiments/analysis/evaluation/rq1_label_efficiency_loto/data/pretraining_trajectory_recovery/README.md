# Recovered seed-1 and seed-2 pretraining trajectories

This directory preserves the unique ignored JSON evidence recovered from the
`prodigy-rq1-cache` and `prodigy-rq1-revised2` worktrees before cleanup.

## Scope

- `pretraining_trajectories.csv` contains 436 paired metric/score observations:
  264 from seed 1 and 172 from seed 2.
- `checkpoint_manifests.csv` contains the eight available pretraining manifests,
  four per seed.
- Every seed-2 retry is retained under its original `attempt` name. There are 64
  overlapping target/split/update keys across attempts, and their values differ;
  none was silently selected as canonical.
- SHA-256 columns and source-relative paths retain file-level provenance.

These are **pretraining trajectories**, not downstream adaptation `result.json`
cells. They do not fill the package's required 96-cell paired label-efficiency
grid and therefore do not support a seed-1/seed-2 label-efficiency conclusion.
The checkpoint paths in the manifests are historical paths, not durable artifact
references, and the manifests do not contain checkpoint hashes.

The CSVs were produced by `../../consolidate_recovered_pretraining.py`, which
verifies that each metric file has a paired score file and that their accuracy
values agree before merging them.
