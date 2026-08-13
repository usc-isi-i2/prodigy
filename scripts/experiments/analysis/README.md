# Experiment analysis

Analyses are grouped by research question. Each named experiment remains a
self-contained leaf with its own findings, scripts, `data/`, and `figures/`.
Setup and launch files remain in `../setup/<name>/`.

## Transfer

### Matrices

- **PRODIGY / neighbor matching**
  - Single-source transfer: [`nm_single_source_matrix`](transfer/matrices/prodigy_nm/single_source/nm_single_source_matrix/), [`nm_single_source_matrix_facebook`](transfer/matrices/prodigy_nm/single_source/nm_single_source_matrix_facebook/)
  - Merged versus single: [`nm_transfer_matrix`](transfer/matrices/prodigy_nm/merged_vs_single/nm_transfer_matrix/), [`nm_covid_midterm`](transfer/matrices/prodigy_nm/merged_vs_single/nm_covid_midterm/)
  - Downstream transfer: [`nm_single_source_downstream`](transfer/matrices/prodigy_nm/downstream/nm_single_source_downstream/)
  - Identity-disjoint control: [`entity_disjoint_eval`](transfer/matrices/prodigy_nm/identity_disjoint/entity_disjoint_eval/)
- **Architecture-controlled NM:** [`icl_arch_matrix`](transfer/matrices/architecture_nm/icl_arch_matrix/) compares PRODIGY, VISION, and GILT.
- **Native-objective architectures:** [`final_core`](transfer/matrices/native_objective/final_core/) compares PRODIGY/NM and SAMGPT/GraphCL.

### Ladders

- **PRODIGY / neighbor matching**
  - Canonical ladder: [`nm_ladder`](transfer/ladders/prodigy_nm/canonical/nm_ladder/)
  - Order and graph-set extensions: [`nm_ladder_order_robustness`](transfer/ladders/prodigy_nm/order_and_graph_set/nm_ladder_order_robustness/), [`nm_ladder_facebook`](transfer/ladders/prodigy_nm/order_and_graph_set/nm_ladder_facebook/)
- **SAMGPT / GraphCL**
  - Native-objective orders: [`samgpt_graphcl_ladder`](transfer/ladders/samgpt_graphcl/canonical_orders/samgpt_graphcl_ladder/)
  - Weak-to-strong order: [`samgpt_weak_to_strong`](transfer/ladders/samgpt_graphcl/weak_to_strong/samgpt_weak_to_strong/)

### Transfer ablations

- Context hops: [`nm_ladder_nhop2`](transfer/ablations/context_hops/nm_ladder_nhop2/)
- Per-source exposure: [`nm_ladder_fixed_exposure_nhop2`](transfer/ablations/source_exposure/nm_ladder_fixed_exposure_nhop2/)
- Interleaved versus sequential sampling: [`nm_ladder_sequential_nhop2`](transfer/ablations/sampling_schedule/nm_ladder_sequential_nhop2/)
- Train/test edge separation: [`nm_ladder_train_test_nhop2`](transfer/ablations/train_test_separation/nm_ladder_train_test_nhop2/)
- Episode sampling and cross-source shortcuts: [`nm_cross_source_shortcut`](transfer/ablations/episode_sampling/nm_cross_source_shortcut/), [`sampling_strat_comparison`](transfer/ablations/episode_sampling/sampling_strat_comparison/)
- PRODIGY encoder: [`nm_ladder_gatv2`](transfer/ablations/prodigy_encoder/nm_ladder_gatv2/)
- Batch construction: [`nm_all9_source_complete_batch`](transfer/ablations/batch_construction/nm_all9_source_complete_batch/)
- Center sampling: [`nm_all9_radius_finalcore`](transfer/ablations/center_sampling/nm_all9_radius_finalcore/)
- Ladder downstream transfer: [`nm_ladder_downstream`](transfer/ablations/downstream/one_hop/nm_ladder_downstream/), [`nm_ladder_downstream_nhop2`](transfer/ablations/downstream/two_hop/nm_ladder_downstream_nhop2/)
- Saturation
  - PRODIGY: [`pretrain_saturation`](transfer/ablations/saturation/prodigy_nm/one_hop/pretrain_saturation/), [`pretrain_saturation_nhop2`](transfer/ablations/saturation/prodigy_nm/two_hop/pretrain_saturation_nhop2/)
  - SAMGPT: [`samgpt_covid_saturation`](transfer/ablations/saturation/samgpt/covid/samgpt_covid_saturation/), [`samgpt_c5_saturation`](transfer/ablations/saturation/samgpt/five_source/samgpt_c5_saturation/), [`samgpt_covid_correctness_ablation`](transfer/ablations/saturation/samgpt/correctness_controls/samgpt_covid_correctness_ablation/)

## Objectives

- Objective lattice and corpus replications: [`multitask_ssl`](objectives/multitask_ssl/multitask_ssl/), [`multitask_ssl_corpora`](objectives/multitask_ssl/multitask_ssl_corpora/)
- Topology versus feature capability: [`topology_feature_ssl`](objectives/topology_vs_features/topology_feature_ssl/), [`feature_ablation`](objectives/topology_vs_features/feature_ablation/)
- Frozen-probe comparison: [`pretrain_probe_matrix`](objectives/probe_matrix/pretrain_probe_matrix/)
- Within-COVID task transfer: [`covid_task_transfer_matrix`](objectives/task_transfer/covid_task_transfer_matrix/)
- Earlier strategy notebooks: [`best_pretrain_strat`](objectives/legacy_strategy_comparisons/best_pretrain_strat/), [`pretrain_strategy_benchmark`](objectives/legacy_strategy_comparisons/pretrain_strategy_benchmark/)

## Graph characterization

- Structural statistics and divergence: [`graph_divergence`](graph_characterization/statistics/graph_divergence/)
- Biography-embedding geometry: [`bio_embedding_geometry`](graph_characterization/feature_geometry/bio_embedding_geometry/)
- Structure/feature coupling: [`path_feature_coupling`](graph_characterization/structure_feature_coupling/path_feature_coupling/)
- Dataset overlap: [`identity_overlap_audit`](graph_characterization/dataset_overlap/identity_overlap_audit/)
- Similarity as a transfer predictor: [`similarity_vs_transfer`](graph_characterization/similarity_vs_transfer/similarity_vs_transfer/), [`similarity_vs_transfer_v2`](graph_characterization/similarity_vs_transfer/similarity_vs_transfer_v2/)

## Evaluation infrastructure

- Prediction-level diagnostics: [`error_audit`](evaluation/error_audit/)
- Static-link evaluator repair: [`slp_evaluator_repair`](evaluation/slp_evaluator_repair/)
- Shared append-only task tables: [`node_classification`](evaluation/shared_task_tables/node_classification/), [`node_regression`](evaluation/shared_task_tables/node_regression/), [`static_link_prediction`](evaluation/shared_task_tables/static_link_prediction/)

## Program and archive

- Cross-experiment syntheses: [`program/cross_experiment_syntheses`](program/cross_experiment_syntheses/)
- Retired and superseded analyses: [`archive`](archive/)
