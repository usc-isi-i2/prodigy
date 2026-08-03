# Cross-experiment syntheses

Write-ups that span several experiments. Per-experiment findings stay in their own
`analysis/<name>/` folder; this folder is only for documents that read across them,
and it is the entry point to the analysis tree.

| doc | covers |
|---|---|
| [`PROGRAM_FINDINGS.md`](PROGRAM_FINDINGS.md) | the whole program (~13 experiments), thrusts A–H. Last consolidated 2026-07-20; **LP claims superseded 2026-07-23**, see its banner. |
| [`NM_MERGED_VS_SINGLE_SUMMARY.md`](NM_MERGED_VS_SINGLE_SUMMARY.md) | merged-vs-single NM across two source pairs: `nm_transfer_matrix`, `nm_covid_midterm` |
| [`NM_CROSS_SOURCE_STUDY.md`](NM_CROSS_SOURCE_STUDY.md) | the cross-source-shortcut study: `nm_cross_source_shortcut`, `sampling_strat_comparison` |

## The analysis tree

| folder | experiment | findings |
|---|---|---|
| `feature_ablation/` | what NM actually uses (noise/permute ablation) | `FINDINGS.md` |
| `graph_divergence/`, `similarity_vs_transfer/` | graph-distance statistics and whether they predict transfer | `FINDINGS.md` |
| `mix_slp_ablation/` | eval-time 2×2 (rewire × permute) on MIX's static LP | `FINDINGS.md` — ⚠️ premise voided by the rescore |
| `multitask_ssl/` | the {NM, CL, FP} objective lattice — singles, pairs, triple | `FINDINGS.md`, `FINDINGS_rescore.md` |
| `multitask_ssl_corpora/` | the rotation arms replicated on the cov and all8 corpora | `FINDINGS.md` — ⚠️ LP sections superseded |
| `nm_covid_midterm/` | size-imbalance / exposure between covid and midterm | `RESULTS.md` |
| `nm_cross_source_shortcut/` | cross-source-probability sweep on episode sampling | `RESULTS.md` |
| `nm_ladder/` | the 8-rung merged-graph interpolation ladder | `RESULTS.md` |
| `nm_ladder_fixed_exposure_nhop2/` | fixed 10k/source ladder with fair two-hop sampling (Orders A/C) | `FINDINGS.md` |
| `nm_ladder_order_robustness/` | the ladder under different graph orders | — (in progress) |
| `nm_single_source_downstream/` | classification/regression transfer of all 8 single-source NM encoders | `FINDINGS.md` |
| `nm_single_source_matrix/` | 8×8 single-source transfer matrix | `FINDINGS.md` |
| `nm_transfer_matrix/` | matched merged-vs-single comparison | `RESULTS.md` |
| `pretrain_probe_matrix/` | frozen-probe matrix over pretraining strategies | `FINDINGS.md` |
| `pretrain_saturation/` | downstream transfer vs pretraining step (`n_hop=1`) | `FINDINGS.md` |
| `pretrain_saturation_nhop2/` | compute-matched two-hop-context saturation replication | `FINDINGS.md` |
| `slp_evaluator_repair/` | the 2026-07-23 static-LP evaluator repair (method; evaluator itself in `scripts/eval/pair_link_*.py`) | `README.md` |
| `topology_feature_ssl/` | topology-vs-feature SSL objectives (E-series) | `FINDINGS.md`, `RESULTS_directed3log.md` |
| `best_pretrain_strat/`, `sampling_strat_comparison/`, `pretrain_strategy_benchmark/`, `node_classification/`, `node_regression/`, `static_link_prediction/`, `covid_task_transfer_matrix/` | earlier notebook-only analyses; no findings file | — |
| `archive/` | retired analyses and superseded work | see its `README.md` |
