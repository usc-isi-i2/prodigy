# Operational deviations

This append-only log qualifies campaign evidence without changing the frozen
scientific protocol.

## 2026-09-13: full-data smoke exercised final test scoring

The two-epoch, seed-0 GPU smoke at revision `db80fe89` exercised the complete
training, checkpoint, selection, test-metric, and repeat-versus-novel output path.
Its test output was inspected only to verify completion and schema. The architecture,
graph policy, optimizer, seeds, stopping rule, selection metric, and production test
policy were already frozen in that revision, and none were changed in response.

The smoke artifact at `/dataMeR1/phil/gfm/ogbl_collab_sage_lp/smoke_seed0` is
smoke-only and excluded from production aggregation. Final findings must nevertheless
disclose that the official test panel was not literally unseen by the operator.
