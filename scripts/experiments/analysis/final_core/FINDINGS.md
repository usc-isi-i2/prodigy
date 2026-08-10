# Final-core frozen-test results

## Status

The final-core evidence package is complete and self-verifying. It contains the
full fixed-test sweep and the later metric-complete specialist replay:

- **837 physical fixed-test cells**: 3 training seeds × 31 physical models × 9
  targets, evaluated at checkpoint 2,500 on 512 frozen test episodes per cell;
- **729 alias-expanded ladder rows**, backed by 675 distinct ladder cells;
- **243 specialist AUC cells**: 3 seeds × 9 sources × 9 targets with accuracy,
  macro F1, and macro one-vs-rest ROC-AUC;
- the nine-target episode fingerprint ledger, original plans and provenance,
  strict aggregate tables, and a per-file SHA-256 manifest.

The AUC production run completed on 2026-08-08 at 16:01:16 UTC. The earlier
fixed-test workers wrote all 837 expected raw cells, but their launcher stopped
before writing the aggregate tables and completion marker. The tables under
`data/fixed_test/summary/` were therefore reconstructed from those raw cells
with `scripts/experiments/setup/final_core/aggregate_fixed_test.py`. That strict
aggregation accepted the grid without missing, extra, duplicate, malformed, or
non-finite cells.

## Integrity checks

`verify_archive.py` independently checks the saved package rather than relying
on a cluster completion marker. It enforces the physical plan, cell counts,
protocol fields, checkpoint and episode counts, finite metrics, fingerprint
ledger, raw-to-summary agreement, matrix dimensions, cross-run replay, and all
file hashes.

The fixed-test and AUC specialist replays use identical planned and observed
episode fingerprints for every target. Of the 243 specialist cells, 233 replay
accuracies are bit-identical. Ten differ by exactly one prediction out of
61,440 query predictions (maximum absolute accuracy difference
`1.627604166667962e-05`); none differs by more. The archive records this
reconciliation explicitly instead of silently treating the two executions as
byte-identical.

Run the complete check from the repository root:

```bash
/opt/homebrew/bin/python3.11 \
  scripts/experiments/analysis/final_core/verify_archive.py
```

Expected terminal line:

```text
FINAL_CORE_ARCHIVE_OK fixed_cells=837 auc_cells=243 files=1114 replay_exact=233/243 sha256=ed8985fedd9fc851ba37aa08bcd34e93105241c899f08dc87d8e4dea7c52edf1
```

## Specialist transfer result

The final three-seed mean ROC-AUC matrix is
`data/auc/summary/single_source_roc_auc_ovr_macro_three_seed_mean.csv`.
Same-graph specialists are best on eight of the nine targets. The exception is
Facebook Page-Page, where the Ukraine/Russia specialist reaches 0.8660 ROC-AUC
versus 0.8127 for the Facebook specialist, a difference of +0.0533. Mean
same-graph ROC-AUC is 0.8858; mean off-diagonal transfer ROC-AUC is 0.7843.

These are descriptive results from three training seeds on one fixed episode
stream. The seeds do not resample evaluation episodes, so their sample standard
deviation measures training-run variation, not an evaluation-episode confidence
interval.

## Archive layout

```text
data/
  manifest.json                 all file sizes, SHA-256 hashes, source paths
  fixed_test/
    physical_plan.tsv           93 checkpoint-seed jobs
    provenance.txt              original run provenance
    results/                    837 immutable raw cell JSON files
    summary/                    strict matrix and ladder aggregates
  auc/
    complete_utc.txt            original production completion marker
    provenance.txt              original AUC replay provenance
    reference/                  frozen episode fingerprint ledger
    results/                    243 metric-complete raw cell JSON files
    summary/                    per-seed and three-seed metric matrices
```

The checkpoint binaries remain in the dedicated Tucker training state root
recorded by both provenance files; they are intentionally not duplicated in Git.
The committed package preserves all compact result evidence needed to inspect,
reaggregate, and verify the reported tables without access to the cluster logs.
