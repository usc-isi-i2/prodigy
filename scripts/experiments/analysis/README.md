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
- **Architecture-controlled NM:** [`icl_arch_matrix`](transfer/matrices/cross_architecture/icl_arch_matrix/) compares PRODIGY, VISION, and GILT.
- **Native-objective architectures:** [`final_core`](transfer/matrices/cross_model/final_core/) compares PRODIGY/NM and SAMGPT/GraphCL.

### Ladders

- **PRODIGY / neighbor matching**
  - Canonical ladder: [`nm_ladder`](transfer/ladders/prodigy_nm/baseline/nm_ladder/)
  - Order and graph-set extensions: [`nm_ladder_order_robustness`](transfer/ladders/prodigy_nm/robustness/nm_ladder_order_robustness/), [`nm_ladder_facebook`](transfer/ladders/prodigy_nm/robustness/nm_ladder_facebook/)
- **SAMGPT / GraphCL**
  - Native-objective orders: [`samgpt_graphcl_ladder`](transfer/ladders/samgpt_graphcl/baseline/samgpt_graphcl_ladder/)
  - Weak-to-strong order: [`samgpt_weak_to_strong`](transfer/ladders/samgpt_graphcl/weak_to_strong/samgpt_weak_to_strong/)

### Ablations by model

- Context hops: [`nm_ladder_nhop2`](transfer/ablations/prodigy_nm/context_depth/nm_ladder_nhop2/)
- Per-source exposure: [`nm_ladder_fixed_exposure_nhop2`](transfer/ablations/prodigy_nm/source_exposure/nm_ladder_fixed_exposure_nhop2/)
- Labeled-source diversity: [`labeled_mixture_diversity_cls500`](transfer/ablations/prodigy_nm/source_diversity/labeled_mixture_diversity_cls500/)
- Interleaved versus sequential sampling: [`nm_ladder_sequential_nhop2`](transfer/ablations/prodigy_nm/source_schedule/nm_ladder_sequential_nhop2/)
- Train/test edge separation: [`nm_ladder_train_test_nhop2`](transfer/ablations/prodigy_nm/split_integrity/nm_ladder_train_test_nhop2/)
- Episode sampling and cross-source shortcuts: [`nm_cross_source_shortcut`](transfer/ablations/prodigy_nm/episode_sampling/nm_cross_source_shortcut/), [`sampling_strat_comparison`](transfer/ablations/prodigy_nm/episode_sampling/sampling_strat_comparison/)
- PRODIGY encoder: [`nm_ladder_gatv2`](transfer/ablations/prodigy_nm/encoder_architecture/nm_ladder_gatv2/)
- Batch construction: [`nm_all9_source_complete_batch`](transfer/ablations/prodigy_nm/batch_construction/nm_all9_source_complete_batch/)
- Center sampling: [`nm_all9_radius_finalcore`](transfer/ablations/prodigy_nm/center_sampling/nm_all9_radius_finalcore/)
- Ladder downstream transfer: [`nm_ladder_downstream`](transfer/ablations/prodigy_nm/downstream/nm_ladder_downstream/), [`nm_ladder_downstream_nhop2`](transfer/ablations/prodigy_nm/downstream/nm_ladder_downstream_nhop2/)
- Saturation
  - PRODIGY: [`pretrain_saturation`](transfer/ablations/prodigy_nm/saturation/pretrain_saturation/), [`pretrain_saturation_nhop2`](transfer/ablations/prodigy_nm/saturation/pretrain_saturation_nhop2/)
  - SAMGPT: [`samgpt_covid_saturation`](transfer/ablations/samgpt_graphcl/saturation/samgpt_covid_saturation/), [`samgpt_c5_saturation`](transfer/ablations/samgpt_graphcl/saturation/samgpt_c5_saturation/), [`samgpt_covid_correctness_ablation`](transfer/ablations/samgpt_graphcl/saturation/samgpt_covid_correctness_ablation/)

## Objectives

- Objective lattice and corpus replications: [`multitask_ssl`](objectives/multitask/multitask_ssl/), [`multitask_ssl_corpora`](objectives/multitask/multitask_ssl_corpora/)
- Topology versus feature capability: [`topology_feature_ssl`](objectives/topology_vs_features/topology_feature_ssl/), [`feature_ablation`](objectives/topology_vs_features/feature_ablation/)
- Frozen-probe comparison: [`pretrain_probe_matrix`](objectives/frozen_probes/pretrain_probe_matrix/)
- Within-COVID task transfer: [`covid_task_transfer_matrix`](objectives/within_dataset_transfer/covid_task_transfer_matrix/)
- Earlier strategy notebooks: [`best_pretrain_strat`](objectives/legacy/best_pretrain_strat/), [`pretrain_strategy_benchmark`](objectives/legacy/pretrain_strategy_benchmark/)

## Graphs

- Structural statistics and divergence: [`graph_divergence`](graphs/structure/graph_divergence/)
- Biography-embedding geometry: [`bio_embedding_geometry`](graphs/features/bio_embedding_geometry/)
- Structure/feature coupling: [`path_feature_coupling`](graphs/structure_features/path_feature_coupling/)
- Dataset overlap: [`identity_overlap_audit`](graphs/overlap/identity_overlap_audit/)
- Similarity as a transfer predictor: [`similarity_vs_transfer`](graphs/transfer_prediction/similarity_vs_transfer/), [`similarity_vs_transfer_v2`](graphs/transfer_prediction/similarity_vs_transfer_v2/)

## Evaluation infrastructure

- Prediction-level diagnostics: [`error_audit`](evaluation/error_audit/)
- Static-link evaluator repair: [`static_link_prediction_repair`](evaluation/static_link_prediction_repair/)
- Shared append-only task tables: [`node_classification`](evaluation/task_tables/node_classification/), [`node_regression`](evaluation/task_tables/node_regression/), [`static_link_prediction`](evaluation/task_tables/static_link_prediction/)

## Synthesis and archive

- Cross-experiment syntheses: [`synthesis/cross_experiment`](synthesis/cross_experiment/)
- Retired and superseded analyses: [`archive`](archive/)
