# Findings: RQ1 label-efficiency leave-one-target-out study

## Status

The canonical downstream analysis remains incomplete. Its protocol requires a
paired 96-cell grid: four held-out targets, four label budgets, scratch versus
pretrained initialization, and three seeds. Recovered seed-1 and seed-2 files do
not contain those downstream `result.json` cells.

## Recovered evidence

The storage audit recovered and consolidated 436 pretraining trajectory points
that had remained ignored in two worktrees:

| Source worktree | Seed | Trajectory rows | Manifests |
|---|---:|---:|---:|
| `prodigy-rq1-cache` | 1 | 264 | 4 |
| `prodigy-rq1-revised2` | 2 | 172 | 4 |

The seed-2 worktree contains several retries. Sixty-four target/split/update keys
overlap across attempts, but all have different recorded values, so the recovery
keeps every attempt rather than choosing a winner after the fact.

## Interpretation

This recovery establishes that seed-1 and seed-2 pretraining ran and preserves
their validation/test learning trajectories. It does **not** establish whether
pretraining improved label efficiency: that claim requires the missing paired
downstream adaptation grid. These data should be treated as provenance and
diagnostic evidence, not as an RQ1 result.

See `data/pretraining_trajectory_recovery/README.md` for counts, caveats, and
file-level provenance.
