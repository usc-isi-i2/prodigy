# Target-performance mechanism audit

This folder contains a descriptive audit of the complete seed-0 singleton,
pair, and leave-one-out results available on 2026-09-06 UTC. It starts a mechanism
study; it does not claim to have identified the cause of transfer.

Run the lightweight analysis locally:

```bash
MPLBACKEND=Agg /opt/homebrew/bin/python3.11 scripts/experiments/analysis/graphs/transfer_prediction/target_performance_mechanisms/analyze_lattice.py
```

Run the small production-sampler reproduction in the local model environment:

```bash
/Users/philipp/miniconda3/envs/prodigy/bin/python scripts/experiments/analysis/graphs/transfer_prediction/target_performance_mechanisms/audit_sampler_index_bias.py
```

## Inputs and provenance

Files under `data/raw` were read from Tucker and copied locally with CRLF normalized
to LF; no cluster artifacts were changed. The audit writes SHA-256 hashes for all
inputs to `data/audit_summary.json`.

- Classification: `/dataMeR1/phil/gfm/prodigy-nm-pairs/log/nm_cls_lattice_20260905/`.
  54 models by five targets, 270 cells. The raw table, completeness receipt,
  checkpoint manifest, and producing revision are retained here.
- Pair NM: `/dataMeR1/phil/gfm/prodigy-nm-pairs/log/nm_pairwise_finalcore_eval/production/bs32/summary/pair_metrics_long.tsv`.
  36 models by nine targets, 324 cells.
- LOO NM: `/dataMeR1/phil/gfm/prodigy-nm-loo/log/nm_leave_one_out_finalcore_eval/production/bs32/summary/loo_heldout_metrics.tsv`.
  Nine models evaluated on their omitted source only, nine cells.
- Singleton NM: seed 0 of the existing
  `../similarity_vs_transfer_v2/data/final_core_auc/specialist_cells_three_seed.csv`.
- Graph descriptors: existing graph-divergence and extended-predictor artifacts.
  These describe the full graphs or earlier samples, not the exact final-core
  training episode stream.

The analysis validates complete source-set combinations, unique cells, finite
metrics, checkpoint/seed/episode contracts and within-target fingerprint agreement.
The classification fingerprint audits centers and labels; identical subgraph
realizations require a stronger context fingerprint in the next replay.

## Outputs

- `classification_target_summary.csv`: target and mixture-size descriptive means/ranges.
- `specialist_scores.csv`: separate NM and classification specialist outcomes.
- `mixture_envelope*.csv`: observed gaps to the best constituent specialist.
  These are predictive comparisons, not equal-exposure synergy estimates.
- `foreign_matched_replacement*.csv`: pair comparisons holding the partner,
  target, source count, and nominal total budget fixed; every source is foreign.
- `descriptor_associations.csv`: nine-donor exploratory correlations.
- `classification_similarity*.csv`: existing distances versus foreign specialist
  rankings on each classification target.
- `sampler_index_audit.json`: a synthetic reproduction using the production
  member-selection method.
- `checkpoint_buffer_audit.json`: read-only CPU inspection of the nine specialist
  terminal checkpoints, including finiteness, learned score scale, and stored
  normalization variances. This is an inventory, not a performance explanation.

No inferential p-values are assigned to this one-training-seed grid. Model-level
training uncertainty remains unmeasured. No all-nine classification reference is
silently mixed in from a different evaluation protocol.

Work performed in `/Users/philipp/projects/gfm/prodigy`, branch
`codex/final-core-three-seed-sync`, starting at `9b5ea03c`.
