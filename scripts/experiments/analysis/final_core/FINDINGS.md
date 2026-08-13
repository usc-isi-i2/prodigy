# Final native-pretext matrix and ladder

## Experiment design

The final experiment asks the same transfer question of two architectures while
respecting each architecture's native self-supervised objective:

- **PRODIGY** is evaluated on neighbor matching;
- **SAMGPT** is evaluated on native GraphCL discrimination.

The raw metric scales are architecture-specific and should not be compared
directly. The shared comparisons are the shapes and directional effects: which
specialists transfer, how performance changes as graphs enter a training
mixture, whether the result depends on graph order, and how stable those effects
are across training seeds.

The complete design contains:

- matrix: 2 architectures × 3 seeds × 9 source graphs × 9 target graphs =
  **486 cells**;
- ladder: 2 architectures × 3 seeds × 3 orders × 9 rungs × 9 target graphs =
  **1,458 cells**;
- total: **1,944 logical result cells**.

Physical training and evaluation can reuse equivalent specialist/rung-1 and
all-nine/rung-9 checkpoints. Logical result counts retain all aliases because
they are the units used in the matrix and ladder analyses.

## Current coverage

| architecture | native pretext | matrix | ladder | training seeds | status |
|---|---|---:|---:|---:|---|
| PRODIGY | neighbor matching | 243/243 | 729/729 | 3/3 | complete |
| SAMGPT | GraphCL discrimination | 81/243 | 243/729 | 1/3 | two seeds pending |
| **Total** | architecture-native | **324/486** | **972/1,458** | — | 1,296/1,944 cells |

The missing 648 cells are exactly the two remaining SAMGPT seeds: 162 matrix
cells and 486 ladder cells. They remain part of the planned completeness
contract, but they do not prevent interpretation of the evidence already in
hand.

## Canonical full result tables

Two generated TSV files provide one stable interface to the complete design:

- `data/results_full_long.tsv` has one row per logical experiment cell and
  compact `train_graphs`/`test_graph` fields;
- `data/results_full_graphwide.tsv` contains the same rows and columns, plus
  nine `train:<graph>` and nine `test:<graph>` indicator columns.

Each file has all **1,944 logical cells**, including the 1,296 observed results
and 648 explicit `pending` rows for the two unrun SAMGPT seeds. A stable
`cell_id` identifies each cell. The provenance columns pin the training and
evaluation repositories, commits, configs and config hashes, run/checkpoint
identity where available, source-result path and source-result key. Matrix and
ladder identity is represented by `component`, `order`, `rung`, `added_graph`,
and the exact training-graph set.

Metrics remain architecture-native. `primary_metric`, `primary_value`, and
`primary_direction` give a common analysis interface, while the `nm_*` and
`graphcl_*` columns preserve the available metric families without pretending
that their raw scales are comparable. Pending rows have blank measurements,
not fabricated values. PRODIGY ladder macro-F1 and ROC-AUC are recovered from
the original worker logs at their printed four-decimal precision; the
`nm_f1_auc_source_precision` column distinguishes those values from the
full-precision specialist AUC replay.

Regenerate both tables with `build_full_results.py`. The full verifier rebuilds
their expected contents from the pinned raw evidence and rejects a stale row,
wrong column, missing design cell, incorrect graph indicator, or unresolved
observed source path.

## Visualizations

The reproducible figure set is under `figures/`, with PNG review copies and PDF
vector versions. `plot_final_results.py` regenerates all twelve figures directly
from `data/results_full_long.tsv`: specialist transfer matrices, target-entry
effects, before/after comparisons, order robustness, PRODIGY seed stability,
complete primary-metric ladder trajectories, PRODIGY ROC-AUC ladder
trajectories, PRODIGY ladder seed ranges, and experiment coverage. `figures/README.md`
documents the intended role of each figure.

For inspection without result-oriented annotations or aggregation across seeds
and orders, `figures/neutral_detailed/` contains a separate 16-figure suite.
`plot_neutral_detailed.py` regenerates its per-seed matrices and per-target,
per-order ladder panels from the same canonical table.

## Headline finding

Across both architectures, a target graph becomes easier under the model's own
pretext when that graph enters the training mixture.

- **PRODIGY:** target-entry neighbor-matching accuracy improves in all 24 of 24
  eligible order/graph transitions in each of the three seeds—72 of 72 seeded
  transitions overall. The mean entry gain is +0.0525 accuracy and the median
  gain is +0.0367.
- **SAMGPT:** target-entry native GraphCL BCE improves in 21 of 24 eligible
  order/graph transitions in the observed seed.

This agreement is more informative than comparing raw PRODIGY accuracy to raw
SAMGPT loss. Two architectures, with different mechanisms and native pretexts,
show the same directional source-inclusion effect across many graph/order
contrasts.

## Additional results

For the three-seed PRODIGY specialist matrix, the same-graph specialist has the
best mean ROC-AUC on eight of nine targets. Facebook Page-Page is the exception:
the Ukraine/Russia specialist reaches 0.8660 versus 0.8127 for the Facebook
specialist. Mean same-graph ROC-AUC is 0.8858, compared with 0.7843 across
off-diagonal transfers.

PRODIGY's mean training-seed standard deviation across reported ladder cells is
0.0036 (median 0.0031; maximum 0.0132), substantially smaller than the typical
target-entry gain. The direction and magnitude of the central ladder effect are
therefore stable across its three completed seeds.

For the observed SAMGPT seed, the specialist-maximum rule is much closer to the
ladder behavior than the specialist mean. On the native loss metric, the
specialist-maximum comparison has Pearson r = 0.917 across the 243 ladder cells.
Probability margin is preferable to raw discrimination accuracy for secondary
analysis because the corruption task places many accuracy cells near ceiling.

## Confidence before the remaining SAMGPT seeds

The two additional SAMGPT seeds are still valuable: they will measure
training-seed variance, allow seed-level uncertainty summaries, and test for an
architecture × seed interaction. Until they run, no SAMGPT three-seed error bar
or formal claim of seed invariance should be reported.

That limitation does not make the current conclusions speculative. Confidence
in the qualitative result is already high because:

1. PRODIGY reproduces the target-entry direction in every one of 72 seeded
   transitions;
2. the effect is large relative to PRODIGY's observed seed variation;
3. SAMGPT agrees in 21 of 24 independent graph-entry contrasts despite using a
   different architecture and pretext;
4. the pattern is distributed across nine targets and three graph orders rather
   than being driven by one dataset.

The expected role of the missing seeds is therefore confirmation and sharper
uncertainty, not discovery of the basic direction. A qualitative reversal is
possible in principle, as with any unfinished replication, but the accumulated
evidence gives little reason to expect one.

## Evidence and integrity

The complete PRODIGY component is stored under `data/prodigy_final_core/`:

- `fixed_test/results/`: 837 distinct physical result JSON files;
- `fixed_test/summary/`: the 729-row alias-expanded ladder and strict matrices;
- `auc/results/`: 243 metric-complete specialist result JSON files;
- `auc/summary/`: per-seed and three-seed accuracy, macro-F1, and ROC-AUC;
- `log_recovered_metrics/physical_metrics.tsv`: all 837 physical fixed-test
  cells' accuracy, macro-F1, and ROC-AUC as printed in the original logs;
- `manifest.json`: per-file sizes, hashes, producing commits, and source paths.

The original fixed-test workers wrote every expected raw cell but stopped before
their final aggregation step. The saved summaries were reconstructed from the
complete raw grid using the repository's strict aggregator, which rejected
missing, extra, duplicate, malformed, or non-finite cells. The AUC replay has
its original completion marker. Planned and observed episode fingerprints match
for every target in both executions.

Of 243 repeated PRODIGY specialist cells, 233 accuracies are bit-identical. The
other ten differ by exactly one prediction out of 61,440 query predictions
(maximum absolute difference `1.627604166667962e-05`); none differs by more.
All 243 log-recovered specialist AUC values agree with the full-precision replay
to the expected four-decimal rounding tolerance. The production, recovery, and
continuation logs jointly provide 837 unique completed physical metric records,
matching the fixed-test archive exactly.

Verify the PRODIGY archive from the repository root:

```bash
/opt/homebrew/bin/python3.11 \
  scripts/experiments/analysis/final_core/verify_prodigy_archive.py
```

The observed SAMGPT seed remains in its existing name-aligned archives rather
than being duplicated. `data/samgpt/observed_seed.json` pins the canonical
matrix, ladder, and derived-analysis files by repository path and SHA-256. The
exports do not carry the training seed field, but their pinned configs identify
it exactly as seed 39. Target-specific `eval_seed` values identify fixed
evaluation views, not training seeds.

`data/coverage.json` is the machine-readable completion ledger for the full
two-architecture design. When the additional SAMGPT seeds run, they can be added
without changing the experiment contract or invalidating the conclusions above.

Verify the full current evidence registry, including the pinned SAMGPT exports,
coverage counts, and both target-entry findings:

```bash
/opt/homebrew/bin/python3.11 \
  scripts/experiments/analysis/final_core/verify_experiment.py
```
