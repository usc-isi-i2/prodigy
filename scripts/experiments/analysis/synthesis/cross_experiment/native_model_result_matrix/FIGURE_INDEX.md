# Figure index

Canonical figures are linked by result family so the matrix can be reviewed
without duplicating artifacts.

## Overall coverage

- [`figures/coverage.png`](figures/coverage.png) and
  [`figures/coverage.pdf`](figures/coverage.pdf)

## SSL→CLS saturation

- PRODIGY: [`pretrain_saturation.png`](../../../transfer/ablations/prodigy_nm/saturation/pretrain_saturation/figures/pretrain_saturation.png)
- PRODIGY final-core: [`classification_auc_trajectories.png`](../../../transfer/ablations/prodigy_nm/center_sampling/nm_all9_radius_finalcore/figures/classification_auc_trajectories.png)
- SAMGPT earlier one-seed checkpoint summary: [`samgpt_downstream_saturation.png`](../figures/samgpt_downstream_saturation.png)
- SAMGPT final-core all-nine:
  [`samgpt_all9_saturation.png`](figures/samgpt_all9_saturation.png) and
  [`samgpt_all9_saturation.pdf`](figures/samgpt_all9_saturation.pdf)
- VISION: [`vision_all9_saturation.png`](figures/vision_all9_saturation.png) and
  [`vision_all9_saturation.pdf`](figures/vision_all9_saturation.pdf)
- GraphSAGE narrow TwiBot full-label trajectory:
  [`graphsage_pilot_v1_twibot_cls_saturation.png`](figures/graphsage_pilot_v1_twibot_cls_saturation.png)
  and [`graphsage_pilot_v1_twibot_cls_saturation.pdf`](figures/graphsage_pilot_v1_twibot_cls_saturation.pdf)
- GraphSAGE matched four-target trajectory:
  [`graphsage_matched_saturation_endpoint.png`](figures/graphsage_matched_saturation_endpoint.png),
  [`graphsage_matched_saturation_full_grid.png`](figures/graphsage_matched_saturation_full_grid.png),
  and [`graphsage_matched_saturation_by_target.png`](figures/graphsage_matched_saturation_by_target.png)

## Cross-SSL and family-native source matrices

- VISION native feature-similarity cross-SSL:
  [`vision_native_cross_ssl_matrix.png`](figures/vision_native_cross_ssl_matrix.png),
  [`vision_native_cross_ssl_matrix.pdf`](figures/vision_native_cross_ssl_matrix.pdf),
  [`vision_native_cross_ssl_trajectory.png`](figures/vision_native_cross_ssl_trajectory.png),
  and [`vision_native_cross_ssl_trajectory.pdf`](figures/vision_native_cross_ssl_trajectory.pdf)
- PRODIGY objective lattice: [`perf_by_model_classification.png`](../../../objectives/multitask/multitask_ssl/figures/perf_by_model_classification.png)
- SAMGPT native GraphCL: [`samgpt_native_graphcl_matrix_numbered.png`](../../../transfer/matrices/cross_model/final_core/figures/samgpt_native_graphcl_matrix_numbered.png)
- Native single-source classification audit: [`native_objective_cls_mean_900_seed0.png`](../../../transfer/matrices/cross_architecture/icl_arch_matrix/figures/native_objective_cls_mean_900_seed0.png)

The last figure includes the excluded supervised GILT source sweep for audit
transparency; its GILT values are not credited as native-SSL evidence.

## Downstream CLS matrices

- PRODIGY specialists: [`single_source_classification_heatmap.png`](../../../transfer/matrices/prodigy_nm/downstream/nm_single_source_downstream/figures/single_source_classification_heatmap.png)
- SAMGPT final-core: [`samgpt_downstream_cls_auc_matrix_numbered.png`](../../../transfer/matrices/cross_model/final_core/figures/samgpt_downstream_cls_auc_matrix_numbered.png)
- Target-supervised MLP and GraphSAGE references: [`supervised_graphsage_comparison.png`](../../../transfer/matrices/cross_architecture/icl_arch_matrix/figures/supervised_graphsage_comparison.png)

## Mixture diversity→CLS

- Cross-family mixture explanation: [`mixture_explanation_model_comparison.png`](../../../transfer/matrices/cross_model/final_core/figures/pngs/mixture_explanation_model_comparison.png)
- SAMGPT weak-to-strong order: [`mixture_weak_to_strong.png`](../../../transfer/ladders/samgpt_graphcl/weak_to_strong/samgpt_weak_to_strong/figures/mixture_weak_to_strong.png)
- PRODIGY downstream ladders: [`classification_mean_entry_change.png`](../../../transfer/ablations/prodigy_nm/downstream/nm_ladder_downstream_nhop2/figures/classification_mean_entry_change.png)

## Adaptation efficiency

- Full optimization curves:
  [`optimization_learning_curves.png`](../../../evaluation/adaptation_efficiency/figures/optimization_learning_curves.png)
- Label efficiency:
  [`label_efficiency.png`](../../../evaluation/adaptation_efficiency/figures/label_efficiency.png)
- Updates to 95% of final performance:
  [`updates_to_95pct.png`](../../../evaluation/adaptation_efficiency/figures/updates_to_95pct.png)

The raw and summarized tables preserve all cells; these are not
best-checkpoint-only figures.
