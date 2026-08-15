# Final native-pretext matrix and ladder

## Experiment design

This experiment asks the same source-composition transfer question of two model
families while preserving each family's native self-supervised objective:

- **PRODIGY** is evaluated on neighbor matching;
- **SAMGPT** is evaluated on native GraphCL discrimination.

The metric scales are family-specific and must not be compared directly. The
cross-family comparison is whether the directional source-inclusion pattern
recurs: does a target become easier under the model's own objective when that
graph enters the training mixture?

The complete design contains:

- matrix: 2 families × 3 seeds × 9 single-source models × 9 targets = **486 cells**;
- ladder: 2 families × 3 seeds × 3 orders × 9 rungs × 9 targets = **1,458 cells**;
- total: **1,944 logical result cells**.

Equivalent specialist/rung-1 and all-nine/rung-9 checkpoints are physically
reused. The logical table retains these aliases because they are separate units
in the matrix and ladder analyses.

## Coverage

| model family | native pretext | matrix | ladder | training seeds | status |
|---|---|---:|---:|---:|---|
| PRODIGY | neighbor matching | 243/243 | 729/729 | 3/3 | complete |
| SAMGPT | GraphCL discrimination | 243/243 | 729/729 | 3/3 | complete |
| **Total** | family-native | **486/486** | **1,458/1,458** | — | **1,944/1,944** |

No planned result cell remains pending.

## Canonical result tables

Two generated TSV files provide one stable interface:

- `data/results_full_long.tsv`: one row per logical experiment cell with compact
  `train_graphs` and `test_graph` fields;
- `data/results_full_graphwide.tsv`: the same rows and metrics plus nine
  `train:<graph>` and nine `test:<graph>` indicator columns.

Both contain all **1,944 observed cells**. A stable `cell_id` identifies each
cell. Provenance columns pin repositories, commits, configs and hashes,
run/checkpoint identity, evaluation protocol, and source-result keys. The tables
also record when runs were completed:

- PRODIGY training dates come from dated run IDs, and each evaluation has its
  exact `created_utc` timestamp from the result JSON;
- all 93 SAMGPT physical training runs completed at
  `2026-08-07T18:53:24Z`, as recorded in their training-manifest paths;
- all 93 SAMGPT terminal-checkpoint evaluations completed on `2026-08-08`,
  verified from their result-file dates on Tucker. The table explicitly marks
  this as date precision rather than inventing an exact timestamp.

Metrics remain native to each model family. `primary_metric`, `primary_value`,
and `primary_direction` provide a common interface, while the `nm_*` and
`graphcl_*` columns preserve the full available metric families. PRODIGY ladder
macro-F1 and ROC-AUC were recovered from original worker logs at printed
four-decimal precision; `nm_f1_auc_source_precision` distinguishes them from
the full-precision specialist replay.

Regenerate the tables with `build_full_results.py`. `verify_experiment.py`
independently rebuilds them from the pinned evidence and rejects stale rows,
missing design cells, bad hashes, missing run dates, incorrect graph indicators,
or unresolved evidence paths.

## Headline result

Both model families show a positive source-inclusion effect on average, but the
strength of the pattern differs materially.

- **PRODIGY:** neighbor-matching accuracy improves in all **72/72** seeded
  target-entry transitions. Mean gain is **+0.0525** accuracy and median gain is
  **+0.0367**.
- **SAMGPT:** GraphCL BCE decreases in **49/72** seeded target-entry transitions.
  The seed-level counts are **17/24**, **14/24**, and **18/24** for seeds 39, 40,
  and 41. Mean BCE reduction is **0.0205**. After averaging each order–target
  transition over seeds, **17/24** transitions improve.
- SAMGPT probability margin gives the same sign count, **49/72**, with mean
  margin gain **+0.00736**.

The defensible cross-family claim is therefore qualitative and aggregate:
adding the target graph tends to improve its native-pretext evaluation in both
families. It is universal and comparatively stable for PRODIGY, but only a
majority pattern for SAMGPT. The completed replication does **not** support the
stronger claim that every architecture or every seed exhibits every individual
transition.

## Additional results

For the three-seed PRODIGY specialist matrix, the same-graph specialist has the
best mean ROC-AUC on eight of nine targets. Facebook Page-Page is the exception:
the Ukraine/Russia specialist reaches 0.8660 versus 0.8127 for the Facebook
specialist. Mean same-graph ROC-AUC is 0.8858, compared with 0.7843 across
off-diagonal transfers.

PRODIGY's mean training-seed standard deviation across reported ladder cells is
0.0036 (median 0.0031; maximum 0.0132), well below the typical target-entry
gain.

Across all 729 three-seed SAMGPT ladder cells, the best available specialist is
a useful but imperfect predictor of mixture behavior. The Pearson correlation
is **0.838** for native loss (MAE **0.00994**) and **0.898** for probability
margin (MAE **0.00315**). Probability margin remains preferable to raw
discrimination accuracy for secondary analysis because many accuracy cells are
near ceiling.

## Paper implication

This experiment is complete enough for the core cross-family source-composition
claim; no additional training is needed to fill its planned grid. The paper
should frame PRODIGY and SAMGPT as model families—architecture plus native
objective—not as a causal architecture ablation. The strongest result is the
PRODIGY effect, with SAMGPT serving as a directionally consistent but weaker
replication and an informative boundary on universality.

## Evidence and integrity

The complete PRODIGY evidence is stored under `data/prodigy_final_core/`:

- 837 distinct physical fixed-test result JSON files;
- 729 alias-expanded ladder rows and strict matrices;
- 243 full-precision specialist AUC replay cells;
- 837 log-recovered physical metric records;
- a manifest with file sizes, hashes, producing commits, and source paths.

The original fixed-test workers wrote every expected raw cell but stopped before
their final aggregation. The saved summaries were reconstructed from the full
grid with the repository's strict aggregator. Of 243 repeated specialist cells,
233 accuracies are bit-identical; the other ten differ by exactly one prediction
out of 61,440, and all AUC values agree to the expected logged precision.

The three-seed SAMGPT evidence is stored under `data/samgpt/three_seed/`. It was
exported from `samgpt-social` branch `codex/samgpt-final-core` and committed there
as `34fc522a66a4afa5c5164a77109df4fa6b392de3`. The local registry pins that
commit, the training and evaluation commits, all imported file hashes, seeds
39/40/41, checkpoints 20/60/180/500, and the run-date audit. The source export
contains 3,348 physical checkpoint–target cells; the canonical experiment table
uses the 837 terminal-checkpoint physical cells to produce 972 SAMGPT logical
matrix and ladder rows.

Verify the complete evidence registry and both canonical tables from the
repository root:

```bash
/opt/homebrew/bin/python3.11 \
  scripts/experiments/analysis/transfer/matrices/cross_model/final_core/verify_experiment.py
```
