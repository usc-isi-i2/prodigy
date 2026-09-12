# Experiment analysis

- [September 11 node-only MLP ladder, activation diagnostics, LeakyReLU screening and Suspended repair](transfer/ablations/node_mlp/node_mlp_ladder_20260911/RESULTS.md)

Analyses are grouped by research question. Each named experiment remains a
self-contained leaf with its own findings, scripts, `data/`, and `figures/`.
Setup and launch files remain in `../setup/<name>/`.

## Transfer

### Matrices

- **PRODIGY / neighbor matching**
  - Single-source transfer: [`nm_single_source_matrix`](transfer/matrices/prodigy_nm/single_source/nm_single_source_matrix/), [`nm_single_source_matrix_facebook`](transfer/matrices/prodigy_nm/single_source/nm_single_source_matrix_facebook/)
  - Social-versus-citation pilot: [`social_specificity_pilot`](transfer/matrices/prodigy_nm/single_source/social_specificity_pilot/)
  - Merged versus single: [`nm_transfer_matrix`](transfer/matrices/prodigy_nm/merged_vs_single/nm_transfer_matrix/), [`nm_covid_midterm`](transfer/matrices/prodigy_nm/merged_vs_single/nm_covid_midterm/)
  - Downstream transfer: [`nm_single_source_downstream`](transfer/matrices/prodigy_nm/downstream/nm_single_source_downstream/)
  - Identity-disjoint control: [`entity_disjoint_eval`](transfer/matrices/prodigy_nm/identity_disjoint/entity_disjoint_eval/)
- **Architecture-controlled NM:** [`icl_arch_matrix`](transfer/matrices/cross_architecture/icl_arch_matrix/) compares PRODIGY, VISION, and GILT.
- **Component-compensation scope:** [`gilt_component_crossover`](graphs/transfer_prediction/gilt_component_crossover/) records the predeclared GILT test and its failure to replicate the PRODIGY direction.
- **Native-objective architectures:** [`final_core`](transfer/matrices/cross_model/final_core/) compares PRODIGY/NM and SAMGPT/GraphCL.
- **Recent all-pairs impact:** [`source_lattice_comparison`](transfer/matrices/cross_model/source_lattice_comparison/) compares classification pair-minus-singleton distributions for PRODIGY/NM, GraphSAGE/LP, GraphSAGE/GraphMAE, and SAMGPT/GraphCL.

### Ladders

- **PRODIGY / neighbor matching**
  - Canonical ladder: [`nm_ladder`](transfer/ladders/prodigy_nm/baseline/nm_ladder/)
  - Order and graph-set extensions: [`nm_ladder_order_robustness`](transfer/ladders/prodigy_nm/robustness/nm_ladder_order_robustness/), [`nm_ladder_facebook`](transfer/ladders/prodigy_nm/robustness/nm_ladder_facebook/)
  - Cross-task budget grid: [`ladder_cross_task_eval`](transfer/ladders/prodigy_nm/cross_task/ladder_cross_task_eval/)
- **SAMGPT / GraphCL**
  - Native-objective orders: [`samgpt_graphcl_ladder`](transfer/ladders/samgpt_graphcl/baseline/samgpt_graphcl_ladder/)
  - Weak-to-strong order: [`samgpt_weak_to_strong`](transfer/ladders/samgpt_graphcl/weak_to_strong/samgpt_weak_to_strong/)

### Ablations by model

- Source-held-out NM intervention campaign: [`nm_interventions_overnight` results](transfer/ablations/prodigy_nm/nm_interventions_overnight/RESULTS.md) — 144 trained models and 1,296 audited NM cells, including the frozen recipe repetitions.

- Context hops: [`nm_ladder_nhop2`](transfer/ablations/prodigy_nm/context_depth/nm_ladder_nhop2/)
- Per-source exposure: [`nm_ladder_fixed_exposure_nhop2`](transfer/ablations/prodigy_nm/source_exposure/nm_ladder_fixed_exposure_nhop2/)
- Labeled-source diversity: [`labeled_mixture_diversity_cls500`](transfer/ablations/prodigy_nm/source_diversity/labeled_mixture_diversity_cls500/)
- Interleaved versus sequential sampling: [`nm_ladder_sequential_nhop2`](transfer/ablations/prodigy_nm/source_schedule/nm_ladder_sequential_nhop2/)
- Train/test edge separation: [`nm_ladder_train_test_nhop2`](transfer/ablations/prodigy_nm/split_integrity/nm_ladder_train_test_nhop2/)
- Episode sampling and cross-source shortcuts: [`nm_cross_source_shortcut`](transfer/ablations/prodigy_nm/episode_sampling/nm_cross_source_shortcut/), [`sampling_strat_comparison`](transfer/ablations/prodigy_nm/episode_sampling/sampling_strat_comparison/), [`nm_ladder_global_nhop2`](transfer/ablations/prodigy_nm/episode_sampling/nm_ladder_global_nhop2/)
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

- Nonzero mini sampling: [`nonzero_mini_sampling_pilot`](graphs/structure/nonzero_mini_sampling_pilot/FINDINGS.md) compares uniform nodes, edge endpoints and short walks against parent degree and component structure.
- Structural statistics and divergence: [`graph_divergence`](graphs/structure/graph_divergence/)
- Biography-embedding geometry: [`bio_embedding_geometry`](graphs/features/bio_embedding_geometry/)
- Political bio provenance and alignment: [`election_covid_alignment_audit`](graphs/features/election_covid_alignment_audit/); operational decision in [`docs/political_bio_provenance_20260911.md`](../../../docs/political_bio_provenance_20260911.md)
- Structure/feature coupling: [`path_feature_coupling`](graphs/structure_features/path_feature_coupling/)
- Dataset overlap: [`identity_overlap_audit`](graphs/overlap/identity_overlap_audit/)
- Similarity as a transfer predictor: [`similarity_vs_transfer`](graphs/transfer_prediction/similarity_vs_transfer/), [`similarity_vs_transfer_v2`](graphs/transfer_prediction/similarity_vs_transfer_v2/)
- Target-performance mechanisms: [fixed-query stage replay and source-sampler audit](graphs/transfer_prediction/target_performance_mechanisms/FINDINGS_REPLAY.md)
- Completed singleton/pair/leave-one-out lattice audit: [`target_performance_mechanisms`](graphs/transfer_prediction/target_performance_mechanisms/)
- Local transfer contrast: [matched examples and support-anchored transfer health](graphs/transfer_prediction/local_transfer_contrast/FINDINGS.md)
- TRACE schedule scaling: [order-only schedule intervention and bounded replay](graphs/transfer_prediction/trace_schedule_scaling/)
- Role-topology follow-up: [degree-preserving rewiring, second-target repair and aggregation audit](graphs/transfer_prediction/target_performance_mechanisms/FINDINGS_ROLE_TOPOLOGY.md)
- Message-content follow-up: [background-message presence versus content and cached input degree/label audit](graphs/transfer_prediction/target_performance_mechanisms/FINDINGS_MESSAGE_CONTENT.md)
- Class-reference decision: [within-episode ranking, final contrast orientation/strength, and the nominated 50k test](graphs/transfer_prediction/target_performance_mechanisms/FINDINGS_CLASS_REFERENCE_DECISION.md)
- Support-value mechanism: [actual key/value swaps, the successful 50k component prediction, and the class-preference counterexample](graphs/transfer_prediction/target_performance_mechanisms/FINDINGS_CLASS_REFERENCE_KV.md)
- Score accounting and examples: [contrast compression, surviving class preference, and the label-interface boundary](graphs/transfer_prediction/target_performance_mechanisms/FINDINGS_CLASS_REFERENCE_ACCOUNTING.md)

## Evaluation infrastructure

- Ladder training throughput: [`ladder_sampling_profile`](evaluation/performance/ladder_sampling_profile/) profiles full-graph CPU preparation and GPU steps.
- Graph startup: [`graph_load_benchmark`](evaluation/performance/graph_load_benchmark/FINDINGS.md) measures persistent CSR caching, mapped loading, and reader concurrency.

- Frozen-encoder label and optimization efficiency: [`adaptation_efficiency`](evaluation/adaptation_efficiency/)
- Prediction-level diagnostics: [`error_audit`](evaluation/error_audit/)
- Static-link evaluator repair: [`static_link_prediction_repair`](evaluation/static_link_prediction_repair/)
- Shared append-only task tables: [`node_classification`](evaluation/task_tables/node_classification/), [`node_regression`](evaluation/task_tables/node_regression/), [`static_link_prediction`](evaluation/task_tables/static_link_prediction/)

## Synthesis and archive

- Paper-vision downstream evidence and RQ1/RQ2 metrics: [`paper_vision_evidence`](synthesis/cross_experiment/paper_vision_evidence/)
- Cross-experiment syntheses: [`synthesis/cross_experiment`](synthesis/cross_experiment/)
- Native-pretext result-matrix coverage: [`native_model_result_matrix`](synthesis/cross_experiment/native_model_result_matrix/)
- Retired and superseded analyses: [`archive`](archive/)
