# Current evaluation ledger

`data/evaluation_ledger.tsv` is the current repository-only evaluation ledger.
It uses one row per metric result and is intended to be the canonical long-form
source for later model-by-test matrices.

## Scope

- Reads metric-bearing `.csv` and `.tsv` files under `scripts/experiments/analysis/**/data/`.
- Includes reported results, baselines, checkpoint trajectories, ablations, and
  the historical static-link-prediction records marked `void_pre_20260723`.
- Deduplicates exact metric records while retaining all source paths and row numbers.
- Does not yet inspect Git history, other branches, or Tucker logs.

When present, `wandb_exports/graph_clip_run_metadata.csv` is also used as an
optional join source. W&B run names and short IDs are matched to repository run
names/URLs, and W&B's UTC `created_at` replaces a name-parsed timestamp for the
matched rows.

## Important fields

- `model_id`, `test_dataset`, `task`, `target`, `checkpoint_step`, `shots`, and
  `seed` identify the evaluation where the source table provides them.
- `run_date` and `run_timestamp` record when the evaluation ran when recoverable
  from a run ID, run directory, or evidence path. `date_source` and
  `date_precision` show how it was recovered; blank dates are intentionally not
  guessed.
- `metric` and `value` are always a single metric/value pair.
- `provenance_quality` flags rows missing an explicit model ID or test dataset.
- `context_json` retains source-specific fields that do not fit the common schema.
- `source_path` and `source_row` provide auditability back to the repository.

## Rebuild

```bash
python3 scripts/experiments/analysis/evaluation/ledger/build_evaluation_ledger.py
```

The output is deliberately additive in scope: later ingestion from Git history,
branches, Tucker, or W&B should add source adapters rather than overwrite the
meaning of existing rows.
