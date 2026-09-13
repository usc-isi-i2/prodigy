# Experiment overview: June–September 2026

**Reading order: research questions → experiment families → individual evidence.**
This is the broad program overview; the older [consolidated findings](synthesis/cross_experiment/PROGRAM_FINDINGS.md)
is a historical July synthesis, not the current program verdict.

Research snapshot: **11 September 2026**; preservation/link audit **12 September 2026**. Coverage is the local `main` working tree, findings
available in other registered local worktrees, related sibling GFM projects, and the
preserved May–June archive.
It includes local uncommitted research notes. It is an evidence inventory, not a
live Tucker job-status audit or a claim that every remote-only run has been recovered.
Dates below describe documented periods; git import dates are not experiment dates.

## Navigation

1. [How the program developed](#1-how-the-program-developed)
2. [Source composition and cross-graph transfer](#2-source-composition-and-cross-graph-transfer)
3. [Objectives, architectures and training budget](#3-objectives-architectures-and-training-budget)
4. [Graph properties and transfer prediction](#4-graph-properties-and-transfer-prediction)
5. [Why transfer succeeds or fails](#5-why-transfer-succeeds-or-fails)
6. [Evaluation validity and research infrastructure](#6-evaluation-validity-and-research-infrastructure)
7. [What the evidence supports now](#7-what-the-evidence-supports-now)
8. [Complete local inventory](#8-complete-local-inventory)
9. [June and earlier archived work](#9-june-and-earlier-archived-work)
10. [Coverage and maintenance](#10-coverage-and-maintenance)

**Status language.** “Results” means a result is recorded, including negative results;
“pilot” means bounded evidence; “artifacts” means files exist but completion has not
been established here; “superseded” identifies conclusions that must not be reused.
A seed count refers to training seeds unless explicitly called an estimator seed.
Fixed episodes, repeated queries, and shared checkpoints are not independent replicates.
NM (neighbor matching), downstream classification, and pairwise link prediction are
separate tasks; their metrics and conclusions must not be interchanged.

## 1. How the program developed

| Period | Research emphasis | Where it led |
|---|---|---|
| June, with May precursors | Initial training runs, COVID/Ukraine transfer, merged models, augmentation, checkpoint trajectories, TwiBot and HK transfer | Controlled single-versus-merged comparisons and explicit sampling controls replaced several early informal comparisons. |
| Late June–July | Source-specialist matrices, graph-addition ladders, feature ablations, objective combinations, downstream probes | Source identity mattered; the LP repair overturned the apparent MIX synergy. Later ladder orders showed that held-out mixture gains can occur. |
| Late July–August | Fair two-hop controls, fixed source exposure, downstream ladders, Facebook, saturation, model-family comparisons, identity controls | Broader and sometimes multi-seed evidence; graph-entry effects were stronger for NM/LP than for classification. |
| Early September | Large intervention campaign, source lattices, allocation/scheduling, prediction and readout mechanisms | Many plausible improvements failed. Separating representations from their learned readout became central. |
| September 8 onward | Corrected NM failure audits, support/reference decomposition, overlap-aware training, direct LP baselines, proxy-A stability and social-to-KG transfer | More precise failure localization and several important negative controls, including results still on separate branches. |

## 2. Source composition and cross-graph transfer

### 2.1 Single sources versus merged sources

| Study | Question / comparison | Recorded outcome and limits |
|---|---|---|
| [nm_transfer_matrix](transfer/matrices/prodigy_nm/merged_vs_single/nm_transfer_matrix/RESULTS.md) | Matched Ukraine/COVID specialists versus merge, late June. | The original cross-domain “single beats merged” inversion did not reproduce under matched architecture, steps and valid few-shot evaluation. One seed. |
| [nm_cross_source_shortcut](transfer/ablations/prodigy_nm/episode_sampling/nm_cross_source_shortcut/RESULTS.md) | Within-source episodes versus proportional merged episodes. | Within-source episodes improve both source targets at both tested budgets; the matched accuracy gains are about 2 points. One seed. |
| [nm_covid_midterm](transfer/matrices/prodigy_nm/merged_vs_single/nm_covid_midterm/RESULTS.md) | Can balancing rescue a tiny source dominated by COVID? | Balanced within-source sampling rescues Midterm, with a tradeoff on COVID. This does not establish uniform sampling as best for every held-out target. |
| [nm_single_source_matrix](transfer/matrices/prodigy_nm/single_source/nm_single_source_matrix/FINDINGS.md) | Eight specialists × eight NM targets. | Strong native specialists; highly asymmetric foreign transfer. Historical one-hop, 40k checkpoint, one seed; cross-dataset does not imply identity-disjoint. |
| [nm_single_source_matrix_facebook](transfer/matrices/prodigy_nm/single_source/nm_single_source_matrix_facebook/FINDINGS.md) | Extend to nine sources/targets. | Facebook is a receptive target for several Twitter specialists and has a very strong own-source specialist. Adds 17 cells to the historical matrix. |
| [nm_single_source_downstream](transfer/matrices/prodigy_nm/downstream/nm_single_source_downstream/FINDINGS.md) | Do specialist rankings carry to classification/regression? | Classification is target-dependent; regression is weak against raw-feature/degree floors. Check regression protocol before reusing historical values. |
| [entity_disjoint_eval](transfer/matrices/prodigy_nm/identity_disjoint/entity_disjoint_eval/FINDINGS.md) | Remove recurring identities from centers, then whole target subgraphs. | Fully induced removal lowers mean NM accuracy from .3192 to .2861; substantial transfer survives. Three seeds, three compatible ID universes; changed target distributions prevent a simple paired causal interpretation. |

### 2.2 Source-addition ladders and mixture lattices

A ladder is a sequence of independently trained source sets. Source-addition order
is not the same intervention as presenting training episodes sequentially within a run.

| Study | Question / comparison | Recorded outcome and limits |
|---|---|---|
| [nm_ladder](transfer/ladders/prodigy_nm/baseline/nm_ladder/RESULTS.md) | Canonical eight-graph inclusion ladder. | A target usually improves sharply when included in pretraining. The first ladder motivated coverage-versus-dilution analysis. |
| [nm_ladder_order_robustness](transfer/ladders/prodigy_nm/robustness/nm_ladder_order_robustness/FINDINGS.md) | Does the mixture exceed its best constituent on a held-out target? | Combination matters: Order C has 20/21 positive residuals, mean +.0449 AUC; A/B are near or below the best constituent on average. This supersedes a universal “no OOD bonus” claim. |
| [nm_ladder_facebook](transfer/ladders/prodigy_nm/robustness/nm_ladder_facebook/FINDINGS.md) | Insert Facebook at rung six. | Facebook entry gains .02284 AUC near the ceiling, with little immediate change to incumbent targets. One seed. |
| [nm_ladder_fixed_exposure_nhop2](transfer/ablations/prodigy_nm/source_exposure/nm_ladder_fixed_exposure_nhop2/FINDINGS.md) | Fix expected exposure per active source. | All measurable entry events improve their own target, mean +.103 NM AUC. The staircase is not solely caused by shrinking per-source exposure. |
| [nm_ladder_downstream](transfer/ablations/prodigy_nm/downstream/nm_ladder_downstream/FINDINGS.md) | Replay historical ladders on downstream tasks. | Repaired static LP: 11/13 positive entry events. Classification and repaired regression lack a comparable clear entry effect. |
| [nm_ladder_downstream_nhop2](transfer/ablations/prodigy_nm/downstream/nm_ladder_downstream_nhop2/FINDINGS.md) | Downstream effects across fair two-hop schedules/exposure/splits. | 39 physical encoders / 40 logical rows: LP entry positive in 19/21 cases; classification in 9/19. One training seed; dependent paired cases. |
| [final_core](transfer/matrices/cross_model/final_core/FINDINGS.md) | Native-objective matrices and three ladder orders, PRODIGY and SAMGPT. | Complete 1,944 logical cells across three seeds. Family-native metrics differ; compare source-inclusion patterns rather than raw scores across families. |
| [samgpt_graphcl_ladder](transfer/ladders/samgpt_graphcl/baseline/samgpt_graphcl_ladder/FINDINGS.md) | SAMGPT native GraphCL inclusion behavior. | Own-target BCE improves in 21/24 entry comparisons; best-specialist rule approximates behavior better than the specialist mean. One-seed historical study. |
| [samgpt_weak_to_strong](transfer/ladders/samgpt_graphcl/weak_to_strong/samgpt_weak_to_strong/FINDINGS.md) | Deliberately add progressively stronger TwiBot donors. | Mixture tracks the best constituent approximately, with small positive residuals; endpoint +.0294 AUC. A target-specific diagnostic, not a universal bound. |
| [nm_pairwise_source_additions](transfer/matrices/prodigy_nm/source_additions/nm_pairwise_source_additions/README.md) | All 36 source pairs versus constituent specialists. | 648 directional NM delta cells, nine targets, seed 0 at 2,500 steps. Separates incumbent, newly added and jointly held-out targets. |
| [source_lattice_comparison](transfer/matrices/cross_model/source_lattice_comparison/README.md) | Compare pair effects across models/objectives. | Classification and task-aligned lattice views for PRODIGY/NM, GraphSAGE/LP, GraphSAGE/GraphMAE and SAMGPT/GraphCL. Protocols and metric scales remain distinct. |
| [labeled_mixture_diversity_cls500](transfer/ablations/prodigy_nm/source_diversity/labeled_mixture_diversity_cls500/RESULTS.md) | Labeled-source diversity at fixed compute, continued to 1k. | Held-out macro classification rises from .7061 at one source to .7475 at four sources at 1k. Gains depend on target/donor; many individual models remain checkpoint-sensitive. |
| [paper_vision_evidence](synthesis/cross_experiment/paper_vision_evidence/FINDINGS.md) | Combine downstream ladder/lattice/schedule evidence. | 1,352 protocol-specific physical cells, with explicit reuse/exclusions. Supports target-dependent benefits, not generally non-degrading pretraining. |

### 2.3 Exposure, scheduling and training interventions

| Study | Question / comparison | Recorded outcome and limits |
|---|---|---|
| [nm_all9_source_complete_batch](transfer/ablations/prodigy_nm/batch_construction/nm_all9_source_complete_batch/FINDINGS.md) | One source-complete batch versus batch-one training. | Equal-episode batch-nine is worse; nine-times exposure nearly ties the mean and substantially harms Election. No sample-efficiency improvement. |
| [nm_all9_radius_finalcore](transfer/ablations/prodigy_nm/center_sampling/nm_all9_radius_finalcore/FINDINGS.md) | Restrict episode class centers to nearby graph regions. | Three seeds: radius mixture does not beat global training on matched close-radius tests and loses on global tests; close-only is worse. |
| [nm_interventions_overnight](transfer/ablations/prodigy_nm/nm_interventions_overnight/RESULTS.md) | Baseline plus 16 interventions over eight rungs, frozen-recipe repetitions. | 144 trained models / 1,296 NM cells completed. Included-source gains generally do not transfer to unseen TwiBot. One exploratory seed; recipe repetitions are not an interaction test. |
| [trace_schedule_scaling](graphs/transfer_prediction/trace_schedule_scaling/FINDINGS.md) | Exactly matched data, blocked/replay/interleaved training. | 27 models, three seeds, 2/3/4 sources. No universal pair-to-large-mixture forgetting transition. Initial TRACE fusion gain needs the stronger readout baseline below. |
| [Schedule-stage correction](graphs/transfer_prediction/trace_schedule_scaling/FINDINGS_STAGE_AUDIT.md) | Does schedule-fusion outperform direct intermediate readout? | U1 equal ensemble beats TRACE in accuracy/F1/AUC on the supported four-target fresh panel; NLL is worse. The original fusion result is not broad superiority. |
| [trace_health_guided](graphs/transfer_prediction/trace_health_guided/FINDINGS.md) | Allocate source exposure using target-support agreement. | Three-seed fresh-stream mean loses to uniform interleaving in accuracy and AUC. No reliable allocation improvement. |
| [LOO exposure × block size](../setup/nm_loo_schedule_signal/RESULTS.md) | 18 runs: proportional/uniform × block sizes 1/16/full × three seeds. | On held-out Ukraine, proportional exposure wins at every block size; k=1 wins for both exposure rules. Predicted uniform-plus-k=1 optimum rejected. All-nine k=1 follow-up: proportional changes macro AUC by −.00304 and costs Ukraine-Suspended .04854; the gain is not universal. |

The two-hop baseline, sequential, split-aware, global/unconfined, GATv2 and
sampling-improvement variants have their own records in §8. Where only an assembler,
notebook or figure is present, this overview does not invent a completed verdict.

## 3. Objectives, architectures and training budget

### 3.1 What should the encoder learn?

| Study | Question / comparison | Recorded outcome and limits |
|---|---|---|
| [feature_ablation](objectives/topology_vs_features/feature_ablation/FINDINGS.md) | Remove, permute or replace input features. | Historical one-hop NM depends strongly on real neighborhood feature content. This is not proof that representations contain no adjacency information. |
| [multitask_ssl](objectives/multitask/multitask_ssl/FINDINGS.md) | Seven-arm NM/CL/FP objective lattice. | Corrected LP: NM .757, NMFP .738, NMCL .679, MIX .680. No emergent three-way LP synergy. MIX retains positive mean regression, but this is one-seed evidence. |
| [Objective corpus rescore](objectives/multitask/multitask_ssl/FINDINGS_rescore.md) | Same 15 frozen checkpoints under valid pairwise LP. | NM > MIX > CL > FP by mean LP in each tested corpus. Use this instead of invalid LP conclusions in the older corpus/rotation write-ups. |
| [topology_feature_ssl](objectives/topology_vs_features/topology_feature_ssl/FINDINGS.md) | Degree inputs, aggregation, augmentation and E4 multi-head objectives. | E1 has positive regression; E4/E4r hurt retained node metrics. Old LP and joint-score rankings are invalid unless specifically rescored. |
| [pretrain_probe_matrix](objectives/frozen_probes/pretrain_probe_matrix/FINDINGS.md) | Frozen pretrained encoders versus raw/untrained floors. | Historical raw features win regression in this benchmark. Old static-LP rankings are void; do not generalize either result to every later model. |
| [mt_transfer_pilot](transfer/matrices/prodigy_mt/mt_transfer_pilot/README.md) | Supervised MT versus equal-compute NM+MT transfer. | Matched 5×5 pilot at 900 updates; older 10-shot NM cannot serve as a matched baseline for its 3-shot comparison. |

### 3.2 How much training, and which model family?

| Study | Question / comparison | Recorded outcome and limits |
|---|---|---|
| [pretrain_saturation](transfer/ablations/prodigy_nm/saturation/pretrain_saturation/FINDINGS.md) | Budget trajectories for three PRODIGY corpora. | Classification gains arrive early, largely by 500 steps. Interpret regression using the repaired probe rather than historical episodic scores. |
| [pretrain_saturation_nhop2](transfer/ablations/prodigy_nm/saturation/pretrain_saturation_nhop2/FINDINGS.md) | Fair one-hop versus two-hop context budgets. | Early classification saturation persists; no consistent two-hop gain. An oversized two-hop pilot was stopped and excluded. |
| [samgpt_covid_saturation](transfer/ablations/samgpt_graphcl/saturation/samgpt_covid_saturation/FINDINGS.md) | SAMGPT single-source convergence. | Transfer is nearly saturated by 500–1,000 updates while native loss continues improving. One seed. |
| [samgpt_c5_saturation](transfer/ablations/samgpt_graphcl/saturation/samgpt_c5_saturation/FINDINGS.md) | SAMGPT five-source convergence. | TwiBot validation peaks at the first saved 200-update checkpoint; improving GraphCL loss does not imply improving transfer. |
| [samgpt_covid_correctness_ablation](transfer/ablations/samgpt_graphcl/saturation/samgpt_covid_correctness_ablation/FINDINGS.md) | Explain high zero-step baseline and audit prompt/view handling. | Real graph structure explains much of the random-GCN baseline. Corrected prompt routing lowers loss but worsens tested transfer; resampling costs more without a useful gain. |
| [icl_arch_matrix](transfer/matrices/cross_architecture/icl_arch_matrix/FINDINGS.md) | PRODIGY/VISION/GILT under matched NM data and low budget. | 372/372 cells at 100 updates, one seed. VISION leads mean classification in this screen; not a converged or parameter-matched architecture ranking. |
| [native_model_result_matrix](synthesis/cross_experiment/native_model_result_matrix/FINDINGS.md) | Audit native-objective family coverage. | Records PRODIGY, VISION and SAMGPT coverage, GILT gaps, and baseline limitations. Its dated snapshot must not override newer individual studies. |
| [nm_pinsage_fast_ablation](graphs/encoders/nm_pinsage_fast_ablation/FINDINGS.md) | PinSAGE versus matched GraphSAGE on exact held-out NM streams. | PinSAGE loses all 16 screened cells. One seed; negative screen does not justify completing the same broad sweep. |
| [adaptation_efficiency](evaluation/adaptation_efficiency/FINDINGS.md) | Frozen encoders across label budgets and update choices. | Complete 48/48 model-target grids; leave-one-target-out update selection. PRODIGY leads the reported mean label-efficiency summary; target/oracle selection is separately labeled. |

### 3.3 Transfer beyond social graphs

| Study | Question / comparison | Recorded outcome and limits |
|---|---|---|
| [social_specificity_pilot](transfer/matrices/prodigy_nm/single_source/social_specificity_pilot/FINDINGS.md) | Twitter/Facebook/Cora/PubMed 4×4 pilot. | Twitter transfers strongly to citation targets; citation-to-social transfer is much weaker. Compare within targets because citation uses 5-way and social 30-way episodes. |
| [public_prodigy_kg](graphs/transfer_prediction/public_prodigy_kg/FINDINGS.md) | Original-style Wiki → FB15K-237 replication and role test. | Native accuracy .73795; removing support or query context sharply hurts. Does not replicate the political support-help/query-harm conjunction. |
| [Social → FB15K-237 adapter](transfer/matrices/prodigy_nm/downstream/social_to_fb15k_adapter/FINDINGS.md) | Four frozen social specialists, 20-way KG task; text versus endpoint flags. | Text-only accuracy 36.05–51.26%, above 5% chance. Replacing two coordinates with role flags is not a general repair. Native KG architecture differs; separate branch, not a causal source-only comparison. |

### 3.4 Related model studies in sibling projects

These are part of the broader GFM program even though their code lives outside
PRODIGY. Copies of the same historical `RESULTS.md` occur in several worktrees;
they are not counted as independent experiments. The experiment-specific records
below take precedence over those inherited summaries.

| Study | Recorded outcome / scope | Evidence |
|---|---|---|
| GraphSAGE held-out mixture scaling | Six-rung held-out ladder is nonmonotonic. Three-seed LOO-versus-target-specialist endpoint varies by target; mean AUC difference −.0056. This is not a pure source-count contrast. | [Original results](/Users/philipp/projects/gfm/mixture-scaling/results/RESULTS.md) |
| Strict GraphSAGE transfer and structural inputs | Follow-up with induced node partitions, source-edge validation, raw/untrained controls and structural-feature comparisons; distinct from the initial fixed-budget ladder. | [Strict study](https://github.com/philippnoah/mixture-scaling/blob/da521b8aec95859ab366bad8723c418273a719c5/results/STRICT_STUDY.md) |
| Social-9 GraphSAGE LP/GraphMAE lattices | 54 models per objective; 1,188 downstream cells completed across classification and repaired LP. Pair effects are summarized with the other model families in §2.2. | [Protocol and completion](https://github.com/philippnoah/mixture-scaling/blob/da521b8aec95859ab366bad8723c418273a719c5/docs/social9_source_lattice.md) |
| SAMGPT Social-9 lattice | 54 models at seed 39. Classification pairs lose .01393 AUC on average versus the better constituent (25.6% wins), while native GraphCL pairs improve. Objective gains do not imply downstream gains. | [Findings](https://github.com/philippnoah/samgpt-social/blob/ab944447e929c34eade772c9a5ee700b6f8c611e/analysis/source_lattice/FINDINGS.md) |
| Balanced BCE and cosine-ranking repair | Four matched seeds on held-out TwiBot transfer. Edge optimization improves, but neither balanced BCE nor cosine ranking improves downstream classification over the legacy objective. | [Completed pilot](/Users/philipp/projects/gfm/mixture-scaling-balanced-pilot/docs/balanced_objective_fast_pilot.md) |
| Node-only MLP FP/LP transfer | Nine specialists under feature prediction and LP, without message passing; designed 81 FP and 54 LP cells. The reviewed protocol alone does not establish completion or a winning result. | [Protocol](/Users/philipp/projects/gfm/mixture-scaling/docs/node_only_transfer.md) |
| Node-only MLP source ladder | Nine source rungs, one seed, fixed 2,500 updates; six LP targets. Included and unseen targets must remain separate. Protocol record; completion not inferred here. | [Protocol](/Users/philipp/projects/gfm/mixture-scaling/docs/node_mlp_ladder.md) |
| Topology-only GIN | Three-seed Ukraine-Suspended control: constant-feature GIN .5707 AUC does not beat log-degree .5764; degree-preserving rewiring changes little. No evidence of a useful beyond-degree topology gain. | [Results](https://github.com/philippnoah/gin-topology/blob/c02f2394dec78800ec82606a68e383483fc4825e/RESULTS.md) |
| GraphGlue source-addition pilot | 21 models / 210 target evaluations, three seeds, Computers 1-shot. Neither addition path is monotonic; baseline parity with the paper is not established, so this is not a refutation of its reported regime. | [Final report](https://github.com/philippnoah/graphglue-results/blob/619e1903712f28a4f5deab876567f08a80c36edb/FINAL_REPORT.md) |
| GraphGlue one-full-epoch ladder | Seven models / 70 paired evaluations, one seed. Baseline 49.68%; official final 48.31%, adversarial final 45.62%. Neither mean trajectory is monotonic. Paper reproduction gap and baseline repeat discrepancy remain unresolved. | [Final one-epoch findings](transfer/ladders/graphglue/graphglue_ladder_oneepoch/FINDINGS.md) |

## 4. Graph properties and transfer prediction

### 4.1 What differs across datasets?

| Study | Question / comparison | Recorded outcome and limits |
|---|---|---|
| [Graph divergence](graphs/structure/graph_divergence/) | Structural statistics, feature marginals and feature/structure coupling. | Foundational descriptive analysis and notebooks; use the graph catalog for current artifact identities/counts. |
| [bio_embedding_geometry](graphs/features/bio_embedding_geometry/FINDINGS.md) | Shared text, embedding concentration and dimensions across nine graphs. | Substantial normalized-bio overlap and different feature geometries. Equal text is not proof of equal account identity. |
| [path_feature_coupling](graphs/structure_features/path_feature_coupling/FINDINGS.md) | Feature differences versus graph path length and local context. | Weak whole-vector distance trends hide predictive low-dimensional structure; mean similarity alone is insufficient. |
| [identity_overlap_audit](graphs/overlap/identity_overlap_audit/FINDINGS.md) | Exact identities versus text proxies. | Large overlap among compatible Twitter ID universes. Missing/incompatible IDs are not zero overlap; induced disjoint evaluation provides the stronger transfer control. |

### 4.2 Can these properties select a good source?

| Study | Question / comparison | Recorded outcome and limits |
|---|---|---|
| [similarity_vs_transfer](graphs/transfer_prediction/similarity_vs_transfer/FINDINGS.md) | Early graph-distance/transfer correlations. | Feature-cloud separability is a stronger directional predictor than degree distance in the small pilot; not a causal mechanism. |
| [similarity_vs_transfer_v2](graphs/transfer_prediction/similarity_vs_transfer_v2/FINDINGS.md) | Nine-source ranking and strict three-seed final-core extension. | Complete 243 specialist cells, self-transfer excluded; evaluates ranking/regret with graph-aware tests rather than treating directed pairs as IID. |
| [Proxy-A stability](https://github.com/usc-isi-i2/prodigy/blob/1eadf1d9c807f58b6fe81547c3b9f1e1b3c2ea5d/scripts/experiments/analysis/graphs/transfer_prediction/proxy_a_seed_stability/FINDINGS.md) | Three estimator seeds; sorted versus uniform sampling; 4k and 40k nodes. | Negative ranking relationship persists; sorted candidate sampling has node-ID bias. Estimator seeds are not new model-training seeds. Separate branch. |
| [Incremental proxy-A prediction](https://github.com/usc-isi-i2/prodigy/blob/1eadf1d9c807f58b6fe81547c3b9f1e1b3c2ea5d/scripts/experiments/analysis/graphs/transfer_prediction/proxy_a_seed_stability/incremental/FINDINGS.md) | Does proxy-A add beyond source/target effects? | Residual association exists but is strongly influenced by Election/Political. Held-out relative RMSE improves; donor-selection regret worsens. Not a universal selector. |
| [local_transfer_contrast](graphs/transfer_prediction/local_transfer_contrast/FINDINGS.md) | Same Facebook queries, stronger/weaker donor checkpoints. | Substantial complementary errors coexist with a clear aggregate source gap. Local case audits motivate mechanisms but do not independently validate a deployment selector. |

The [random graph-label null control](https://github.com/philippnoah/proxy-a-null-20260909/blob/b131a936705ef575be9907f4b7d1ac1e8a9d3e02/FINDINGS.md)
adds 108 randomized-label and 108 matched real-label fits: held-out accuracy is
50.09% for random labels versus 82.89% for real graph identity. High dimensionality
alone does not explain the real signal; dataset artifacts and node dependence remain
possible. These are estimator controls, not new transfer-training replications.

## 5. Why transfer succeeds or fails

This is a hierarchy of linked studies, not a collection of independent confirmations.
The classification mechanism campaign and the later NM anchor-matching audit use
different tasks and should remain separate.

### 5.1 Downstream classification: target signal, support context and readout

| Study family | What was tested | Current reading |
|---|---|---|
| [Lattice decomposition](graphs/transfer_prediction/target_performance_mechanisms/FINDINGS.md) | 54 models, five classification targets; accompanying NM lattice. | Target means dominate raw classification variation, but meaningful source-by-target differences remain. Descriptive decomposition, not causal variance. |
| [Stage replay and controlled retraining](graphs/transfer_prediction/target_performance_mechanisms/FINDINGS_REPLAY.md) | Available feature/context signal, member selection, coverage and readout constraints. | Coverage prediction fails across seeds; readout-training consistency prediction also fails. Numerical control drift is documented rather than relabeled as replication. |
| [Support versus query context](graphs/transfer_prediction/target_performance_mechanisms/FINDINGS_ROLE_CONTEXT.md) | 720 replay cells plus three-seed political contrast, queries fixed. | Suppressing HK support edges improves political AUC in six seed/stream comparisons; changed class references exactly reproduce altered logits. Localizes a pathway, not its training cause. |
| [Annotation-cue control](graphs/transfer_prediction/target_performance_mechanisms/FINDINGS_ANNOTATION_CUES.md) | Queries without detected published annotation cues. | Support-side ranking effect persists. Does not establish independent ground truth or cue-free supports. |
| [Natural support replacement](graphs/transfer_prediction/target_performance_mechanisms/FINDINGS_NATURAL_SUPPORT.md) / [same-identity context](graphs/transfer_prediction/target_performance_mechanisms/FINDINGS_FIXED_SUPPORT_CONTEXT.md) | Replace valid supports or resample their neighborhoods. | Greater HK political sensitivity persists; resampling the same supports gives essentially no useful AUC repair. |
| [Topology controls](graphs/transfer_prediction/target_performance_mechanisms/FINDINGS_ROLE_TOPOLOGY.md) / [message content](graphs/transfer_prediction/target_performance_mechanisms/FINDINGS_MESSAGE_CONTENT.md) | Rewire while preserving degrees, alter message content/aggregation. | The tested content/topology changes do not reproduce deletion. Message-path presence and learned processing matter; no general “topology irrelevant” conclusion. |
| [Support key/value intervention](graphs/transfer_prediction/target_performance_mechanisms/FINDINGS_CLASS_REFERENCE_KV.md) | Hold attention and query vectors fixed, replace transported support values. | Supports a value-path contribution to political ranking. Later ranking improvements can coexist with near-total class collapse in accuracy. |
| [Label-interface test](graphs/transfer_prediction/target_performance_mechanisms/FINDINGS_LABEL_INTERFACE_DECISION.md) / [score accounting](graphs/transfer_prediction/target_performance_mechanisms/FINDINGS_CLASS_REFERENCE_ACCOUNTING.md) | Initialization sensitivity, contrast compression and residual class preference. | Late suppression effects depend on the label interface. The same headline intervention can have different explanations at different checkpoints. |
| [Matched trajectory synthesis](graphs/transfer_prediction/target_performance_mechanisms/FINDINGS_PAPER_SYNTHESIS.md) | Page-target early/late native versus fixed-prototype readouts. | Both paths improve; suppression catches up under native inference while prototype preference need not reverse. Avoid interpreting relative crossover as absolute deterioration. |
| [Cross-family source ranks](graphs/transfer_prediction/target_performance_mechanisms/FINDINGS_CROSS_MODEL.md) / [gilt_component_crossover](graphs/transfer_prediction/gilt_component_crossover/FINDINGS.md) | Source usefulness across trained families; predeclared GILT crossover. | Source ranks can reverse across families/readouts. The nominated GILT compensation pattern did not replicate. |
| [Public KG boundary](graphs/transfer_prediction/target_performance_mechanisms/FINDINGS_PUBLIC_BOUNDARY.md) | Useful public model with paired support/query context removal. | Both removals harm accuracy, and train-mode normalization couples roles. The private fixed-query pathway does not automatically transfer to this setting. |

### 5.2 Which attempted repairs survive stronger baselines?

| Study | Recorded decision | What it does not establish |
|---|---|---|
| [support_boundary_repair](graphs/transfer_prediction/support_boundary_repair/FINDINGS.md) | Close the tested support-calibration/value-replacement direction: it does not beat plain intermediate ridge. | A gain against a weak native baseline is not sufficient evidence of a useful repair. |
| [encoder_solver_isolation](graphs/transfer_prediction/encoder_solver_isolation/FINDINGS.md) | Completed eight-arm pilot fails its AUC advance criterion. | Does not rule out every gradient-isolation protocol. |
| [centered_ridge_training](graphs/transfer_prediction/centered_ridge_training/FINDINGS.md) | Strengthened direct centered-ridge training closes the earlier utility gap under the same deployment readout. | The prior isolation pilot cannot establish uniquely beneficial native-inference training. |
| [trace_health_guided](graphs/transfer_prediction/trace_health_guided/FINDINGS.md) | Health-guided source allocation does not improve the primary uniform baseline. | A descriptive health signal need not yield a successful allocation intervention. |

### 5.3 Native NM failure audit: from episodes to class references

| Study family | What was tested | Current reading |
|---|---|---|
| [Corrected canonical NM audit](evaluation/error_audit/FINDINGS_NM_CANONICAL_SPLIT.md) | Ukraine/HK native and foreign specialists on held-out edge plans. | Native advantage survives node weighting; the earlier full-graph HK ranking reversal does not. Shared failures remain large. |
| [Complete inputs](evaluation/error_audit/FINDINGS_NM_COMPLETE_INPUTS.md) / [difficulty](evaluation/error_audit/FINDINGS_NM_DIFFICULTY.md) | Entire sampled neighborhoods, degree, repeats and episode grouping. | Neighborhood geometry separates success/failure better than center text, but these are associations. High-degree repeated queries and whole anchor classes are difficult. |
| [Support resampling](evaluation/error_audit/FINDINGS_NM_SUPPORT_RESAMPLING.md) / [support extremes](evaluation/error_audit/FINDINGS_NM_HK_SUPPORT_EXTREMES.md) | Fixed queries, alternative supports versus same-ID context. | Some failures are conditionally rescuable; replacement breaks many correct cases. Nearest-support replacement remains worse than originals on the selected balanced sample. |
| [Native/foreign stage comparison](evaluation/error_audit/FINDINGS_NM_SOURCE_STAGES.md) / [bounded mechanism and repair](evaluation/error_audit/FINDINGS_NM_HK_GOAL.md) | Same encoded inputs under common cosine and native readouts. | The encoder and readout both matter; native readout supplies net gains as well as errors. Local value-path evidence is not a benchmark improvement. |
| [HK failure hierarchy](evaluation/error_audit/FINDINGS_NM_FAILURE_HIERARCHY.md) | Follow first failure through full episode and full canonical stream. | Positive-support terms often favor wrong rivals, but this also occurs in Ukraine; it is not an HK-specific training explanation. Separate worktree. |
| [Multi-positive evaluation](evaluation/error_audit/FINDINGS_NM_HK_OVERLAP_AWARE.md) | Count any held-out-neighbor candidate as valid for an LP interpretation. | Unchanged HK predictions rise from 17.86% assigned-anchor to 23.26% multi-positive accuracy. This is a metric change, not model improvement. |
| [Matched overlap-aware training](evaluation/error_audit/FINDINGS_NM_HK_OVERLAP_TRAINING.md) | Same init and realized episodes; suppress contradicted training-negative messages. | One-seed terminal treatment loses 1.79 points multi-positive accuracy and worsens MRR. Early small gains reverse; the narrow repair fails. |
| [Broader source-stage audit](evaluation/error_audit/FINDINGS_NM_BROAD_SOURCE_STAGES.md) | Four specialists × four targets, published randomized fixed-test episodes. | Extends native/foreign localization. Different member-selection protocol from detailed lowest-sorted HK audit; do not pool the streams. |
| [HK direct-link baselines](evaluation/lp_baselines/hk_lp_baselines/FINDINGS.md) | Direct LP on identical degree-matched held-out pairs. | Frozen NM cosine .5550 AUC trails LP-trained MLP .6762 and GraphSAGE .6849. NM anchor matching and direct LP remain distinct benchmarks; one-seed pilot. |

The full list of supporting mechanism, numerical, case, margin, transport and
reference analyses is below. It preserves failed predictions and explanatory
substudies without promoting each diagnostic to an independent experiment.

## 6. Evaluation validity and research infrastructure

| Workstream | Why it matters | Evidence |
|---|---|---|
| Static LP repair | Endpoint-blind episodic scores, random prototypes and degree-confounded negatives invalidated early LP rankings. Use pair-conditioned scoring with common pairs/floors. | [Repair](evaluation/static_link_prediction_repair/), [rescore](objectives/multitask/multitask_ssl/FINDINGS_rescore.md) |
| Regression repair | Some episodic regression evaluations used an untrained head; repaired frozen ridge changes their interpretation. A broad LP-only correction is not sufficient for all historical tables. | [Protocol](../setup/regression_probe_repair/), [repaired downstream ladder](transfer/ablations/prodigy_nm/downstream/nm_ladder_downstream/FINDINGS.md) |
| NM split/member-policy audit | Full-graph diagnostics are not held-out-edge evidence. Historical sorted roles and later randomized roles produce different episode tasks. | [Canonical correction](evaluation/error_audit/FINDINGS_NM_CANONICAL_SPLIT.md), [mechanism replay](graphs/transfer_prediction/target_performance_mechanisms/FINDINGS_REPLAY.md) |
| Deterministic numerical controls | Same seed and complete input stream did not guarantee matching default training outcomes; a specific repeated-index backward path was localized. | [Numerics](graphs/transfer_prediction/target_performance_mechanisms/FINDINGS_NUMERICS.md) |
| Checkpoint and fixed-stream ledgers | Nominal budgets can exceed saved steps; aliases and shared endpoints inflate logical counts unless deduplicated. | [Ledger](evaluation/ledger/), [final-core](transfer/matrices/cross_model/final_core/FINDINGS.md) |
| Loader/shared-graph throughput | CPU subgraph construction bottlenecks training; shared-graph concurrency was smoke-validated, not scientifically evaluated. | [Profile](evaluation/performance/ladder_sampling_profile/FINDINGS.md), [fast training](../../../docs/fast_training.md) |
| Graph/task construction | TwiBot, HK, Facebook/citation/KG inputs and node/edge tasks enable later studies; these are not performance wins themselves. | [Graph catalog](../../../docs/graph_catalog.json), setup inventory below |

Temporal LP remains unrescored in the reviewed validity record. Do not use its
historical values as current evidence. For all tasks, distinguish across-episode
variation from across-training-seed variation and use the exact referenced protocol.

## 7. What the evidence supports now

1. **Composition matters more than source count alone.** Coverage effects are strong
   in NM and often LP; downstream classification benefits depend on the particular
   donors, target, task and readout. Later held-out gains rule out July's universal
   “merging never helps OOD” statement.
2. **Early saturation is real in several tested settings.** Improving pretext loss
   or multiplying exposure does not reliably improve downstream transfer.
3. **The original MIX LP breakthrough was an evaluator artifact.** Corrected
   objective results favor NM for LP; later direct LP baselines also show where
   frozen NM remains weak.
4. **Source properties predict part of transfer but do not yet explain it causally.**
   Proxy-A is useful descriptive evidence; incremental prediction and donor
   selection results are more qualified than its raw correlation suggests.
5. **Representation quality and learned inference must be separated.** Support
   context can change class references in useful or harmful ways; several apparent
   repairs lose to a simpler intermediate readout, and the public KG case has a
   different role dependence.
6. **HK difficulty is not explained by one shortcut or one bad module.** Membership
   ambiguity, weak separation and reference construction coexist. Overlap-aware
   message deletion failed as a sufficient training repair.
7. **Evidence strength has improved since July.** Some major designs have three
   training seeds and exact input audits, while many diagnostics remain single-seed
   or selected-case studies. Neither “everything is one seed” nor “everything is
   replicated” is accurate.

## 8. Complete local inventory

The inventory separates experiment/setup families from supporting evidence pages.
Presence of a setup, output directory or `FINDINGS` file does not certify a successful
or fully completed experiment. The summaries above give the reviewed interpretation;
source documents retain exact metrics, protocols, caveats and artifact provenance.

### 8.1 Every top-level setup family

Every directory directly under `scripts/experiments/setup/` is included, even when
it has no matching analysis leaf. Nested variants remain under their owning setup.
“Result note” is a documentation status, not an automatic scientific endorsement.

#### Transfer

| Setup family | Local evidence | Documentation status |
|---|---|---|
| [entity_disjoint_eval](../setup/entity_disjoint_eval/) | [transfer/matrices/prodigy_nm/identity_disjoint/entity_disjoint_eval](transfer/matrices/prodigy_nm/identity_disjoint/entity_disjoint_eval/) | Result note present |
| [final_core](../setup/final_core/) | [transfer/matrices/cross_model/final_core](transfer/matrices/cross_model/final_core/) | Result note present |
| [icl_arch_matrix](../setup/icl_arch_matrix/) | [transfer/matrices/cross_architecture/icl_arch_matrix](transfer/matrices/cross_architecture/icl_arch_matrix/) | Result note present |
| [labeled_mixture_diversity_cls500](../setup/labeled_mixture_diversity_cls500/) | [transfer/ablations/prodigy_nm/source_diversity/labeled_mixture_diversity_cls500](transfer/ablations/prodigy_nm/source_diversity/labeled_mixture_diversity_cls500/) | Result note present |
| [ladder_cross_task_eval](../setup/ladder_cross_task_eval/) | [transfer/ladders/prodigy_nm/cross_task/ladder_cross_task_eval](transfer/ladders/prodigy_nm/cross_task/ladder_cross_task_eval/) | Result note present |
| [mt_transfer_pilot](../setup/mt_transfer_pilot/) | [transfer/matrices/prodigy_mt/mt_transfer_pilot](transfer/matrices/prodigy_mt/mt_transfer_pilot/) | Analysis artifacts; completion not inferred |
| [nm_all9_radius_finalcore](../setup/nm_all9_radius_finalcore/) | [transfer/ablations/prodigy_nm/center_sampling/nm_all9_radius_finalcore](transfer/ablations/prodigy_nm/center_sampling/nm_all9_radius_finalcore/) | Result note present |
| [nm_all9_source_complete_batch](../setup/nm_all9_source_complete_batch/) | [transfer/ablations/prodigy_nm/batch_construction/nm_all9_source_complete_batch](transfer/ablations/prodigy_nm/batch_construction/nm_all9_source_complete_batch/) | Result note present |
| [nm_covid_midterm](../setup/nm_covid_midterm/) | [transfer/matrices/prodigy_nm/merged_vs_single/nm_covid_midterm](transfer/matrices/prodigy_nm/merged_vs_single/nm_covid_midterm/) | Result note present |
| [nm_cross_source_shortcut](../setup/nm_cross_source_shortcut/) | [transfer/ablations/prodigy_nm/episode_sampling/nm_cross_source_shortcut](transfer/ablations/prodigy_nm/episode_sampling/nm_cross_source_shortcut/) | Result note present |
| [nm_interventions_overnight](../setup/nm_interventions_overnight/) | [transfer/ablations/prodigy_nm/nm_interventions_overnight](transfer/ablations/prodigy_nm/nm_interventions_overnight/) | Result note present |
| [nm_ladder_downstream](../setup/nm_ladder_downstream/) | [transfer/ablations/prodigy_nm/downstream/nm_ladder_downstream](transfer/ablations/prodigy_nm/downstream/nm_ladder_downstream/) | Result note present |
| [nm_ladder_downstream_nhop2](../setup/nm_ladder_downstream_nhop2/) | [transfer/ablations/prodigy_nm/downstream/nm_ladder_downstream_nhop2](transfer/ablations/prodigy_nm/downstream/nm_ladder_downstream_nhop2/) | Result note present |
| [nm_ladder_facebook](../setup/nm_ladder_facebook/) | [transfer/ladders/prodigy_nm/robustness/nm_ladder_facebook](transfer/ladders/prodigy_nm/robustness/nm_ladder_facebook/) | Result note present |
| [nm_ladder_fixed_exposure_nhop2](../setup/nm_ladder_fixed_exposure_nhop2/) | [transfer/ablations/prodigy_nm/source_exposure/nm_ladder_fixed_exposure_nhop2](transfer/ablations/prodigy_nm/source_exposure/nm_ladder_fixed_exposure_nhop2/) | Result note present |
| [nm_ladder_gatv2](../setup/nm_ladder_gatv2/) | [transfer/ablations/prodigy_nm/encoder_architecture/nm_ladder_gatv2](transfer/ablations/prodigy_nm/encoder_architecture/nm_ladder_gatv2/) | Analysis artifacts; completion not inferred |
| [nm_ladder_global_nhop2](../setup/nm_ladder_global_nhop2/) | [transfer/ablations/prodigy_nm/episode_sampling/nm_ladder_global_nhop2](transfer/ablations/prodigy_nm/episode_sampling/nm_ladder_global_nhop2/) | Analysis artifacts; completion not inferred |
| [nm_ladder_nhop2](../setup/nm_ladder_nhop2/) | [transfer/ablations/prodigy_nm/context_depth/nm_ladder_nhop2](transfer/ablations/prodigy_nm/context_depth/nm_ladder_nhop2/) | Analysis artifacts; completion not inferred |
| [nm_ladder_sequential_nhop2](../setup/nm_ladder_sequential_nhop2/) | [transfer/ablations/prodigy_nm/source_schedule/nm_ladder_sequential_nhop2](transfer/ablations/prodigy_nm/source_schedule/nm_ladder_sequential_nhop2/) | Analysis artifacts; completion not inferred |
| [nm_ladder_train_test_nhop2](../setup/nm_ladder_train_test_nhop2/) | [transfer/ablations/prodigy_nm/split_integrity/nm_ladder_train_test_nhop2](transfer/ablations/prodigy_nm/split_integrity/nm_ladder_train_test_nhop2/) | Analysis artifacts; completion not inferred |
| [nm_single_source_downstream](../setup/nm_single_source_downstream/) | [transfer/matrices/prodigy_nm/downstream/nm_single_source_downstream](transfer/matrices/prodigy_nm/downstream/nm_single_source_downstream/) | Result note present |
| [nm_single_source_matrix](../setup/nm_single_source_matrix/) | [transfer/matrices/prodigy_nm/single_source/nm_single_source_matrix](transfer/matrices/prodigy_nm/single_source/nm_single_source_matrix/) | Result note present |
| [nm_single_source_matrix_facebook](../setup/nm_single_source_matrix_facebook/) | [transfer/matrices/prodigy_nm/single_source/nm_single_source_matrix_facebook](transfer/matrices/prodigy_nm/single_source/nm_single_source_matrix_facebook/) | Result note present |
| [nm_transfer_matrix](../setup/nm_transfer_matrix/) | [transfer/matrices/prodigy_nm/merged_vs_single/nm_transfer_matrix](transfer/matrices/prodigy_nm/merged_vs_single/nm_transfer_matrix/) | Result note present |
| [pretrain_saturation_nhop2](../setup/pretrain_saturation_nhop2/) | [transfer/ablations/prodigy_nm/saturation/pretrain_saturation_nhop2](transfer/ablations/prodigy_nm/saturation/pretrain_saturation_nhop2/) | Result note present |
| [rq1_native_cls_pilot](../setup/rq1_native_cls_pilot/) | [transfer/rq1_native_cls_pilot](transfer/rq1_native_cls_pilot/) | Analysis artifacts; completion not inferred |
| [samgpt_c5_saturation](../setup/samgpt_c5_saturation/) | [transfer/ablations/samgpt_graphcl/saturation/samgpt_c5_saturation](transfer/ablations/samgpt_graphcl/saturation/samgpt_c5_saturation/) | Result note present |
| [samgpt_covid_correctness_ablation](../setup/samgpt_covid_correctness_ablation/) | [transfer/ablations/samgpt_graphcl/saturation/samgpt_covid_correctness_ablation](transfer/ablations/samgpt_graphcl/saturation/samgpt_covid_correctness_ablation/) | Result note present |
| [samgpt_covid_saturation](../setup/samgpt_covid_saturation/) | [transfer/ablations/samgpt_graphcl/saturation/samgpt_covid_saturation](transfer/ablations/samgpt_graphcl/saturation/samgpt_covid_saturation/) | Result note present |
| [samgpt_graphcl_ladder](../setup/samgpt_graphcl_ladder/) | [transfer/ladders/samgpt_graphcl/baseline/samgpt_graphcl_ladder](transfer/ladders/samgpt_graphcl/baseline/samgpt_graphcl_ladder/) | Result note present |
| [samgpt_weak_to_strong](../setup/samgpt_weak_to_strong/) | [transfer/ladders/samgpt_graphcl/weak_to_strong/samgpt_weak_to_strong](transfer/ladders/samgpt_graphcl/weak_to_strong/samgpt_weak_to_strong/) | Result note present |
| [social_specificity_pilot](../setup/social_specificity_pilot/) | [transfer/matrices/prodigy_nm/single_source/social_specificity_pilot](transfer/matrices/prodigy_nm/single_source/social_specificity_pilot/) | Result note present |

#### Objectives

| Setup family | Local evidence | Documentation status |
|---|---|---|
| [covid_task_transfer_matrix](../setup/covid_task_transfer_matrix/) | [objectives/within_dataset_transfer/covid_task_transfer_matrix](objectives/within_dataset_transfer/covid_task_transfer_matrix/) | Analysis artifacts; completion not inferred |
| [feature_ablation](../setup/feature_ablation/) | [objectives/topology_vs_features/feature_ablation](objectives/topology_vs_features/feature_ablation/) | Result note present |
| [multitask_ssl_corpora](../setup/multitask_ssl_corpora/) | [objectives/multitask/multitask_ssl_corpora](objectives/multitask/multitask_ssl_corpora/) | Result note present |
| [pretrain_probe_matrix](../setup/pretrain_probe_matrix/) | [objectives/frozen_probes/pretrain_probe_matrix](objectives/frozen_probes/pretrain_probe_matrix/) | Result note present |
| [pretrain_strategy_benchmark](../setup/pretrain_strategy_benchmark/) | [objectives/legacy/pretrain_strategy_benchmark](objectives/legacy/pretrain_strategy_benchmark/) | Analysis artifacts; completion not inferred |
| [topology_feature_ssl](../setup/topology_feature_ssl/) | [objectives/topology_vs_features/topology_feature_ssl](objectives/topology_vs_features/topology_feature_ssl/) | Result note present |

#### Graphs

| Setup family | Local evidence | Documentation status |
|---|---|---|
| [centered_ridge_training](../setup/centered_ridge_training/) | [graphs/transfer_prediction/centered_ridge_training](graphs/transfer_prediction/centered_ridge_training/) | Result note present |
| [encoder_solver_isolation](../setup/encoder_solver_isolation/) | [graphs/transfer_prediction/encoder_solver_isolation](graphs/transfer_prediction/encoder_solver_isolation/) | Result note present |
| [gilt_component_crossover](../setup/gilt_component_crossover/) | [graphs/transfer_prediction/gilt_component_crossover](graphs/transfer_prediction/gilt_component_crossover/) | Result note present |
| [graph_divergence](../setup/graph_divergence/) | [graphs/structure/graph_divergence](graphs/structure/graph_divergence/) | Analysis artifacts; completion not inferred |
| [identity_overlap_audit](../setup/identity_overlap_audit/) | [graphs/overlap/identity_overlap_audit](graphs/overlap/identity_overlap_audit/) | Result note present |
| [local_transfer_contrast](../setup/local_transfer_contrast/) | [graphs/transfer_prediction/local_transfer_contrast](graphs/transfer_prediction/local_transfer_contrast/) | Result note present |
| [nm_pinsage_fast_ablation](../setup/nm_pinsage_fast_ablation/) | [graphs/encoders/nm_pinsage_fast_ablation](graphs/encoders/nm_pinsage_fast_ablation/) | Result note present |
| [public_prodigy_kg](../setup/public_prodigy_kg/) | [graphs/transfer_prediction/public_prodigy_kg](graphs/transfer_prediction/public_prodigy_kg/) | Result note present |
| [similarity_vs_transfer](../setup/similarity_vs_transfer/) | [graphs/transfer_prediction/similarity_vs_transfer](graphs/transfer_prediction/similarity_vs_transfer/) | Result note present |
| [similarity_vs_transfer_v2](../setup/similarity_vs_transfer_v2/) | [graphs/transfer_prediction/similarity_vs_transfer_v2](graphs/transfer_prediction/similarity_vs_transfer_v2/) | Result note present |
| [target_performance_mechanisms](../setup/target_performance_mechanisms/) | [graphs/transfer_prediction/target_performance_mechanisms](graphs/transfer_prediction/target_performance_mechanisms/) | Result note present |
| [trace_health_guided](../setup/trace_health_guided/) | [graphs/transfer_prediction/trace_health_guided](graphs/transfer_prediction/trace_health_guided/) | Result note present |
| [trace_schedule_scaling](../setup/trace_schedule_scaling/) | [graphs/transfer_prediction/trace_schedule_scaling](graphs/transfer_prediction/trace_schedule_scaling/) | Result note present |

#### Evaluation

| Setup family | Local evidence | Documentation status |
|---|---|---|
| [adaptation_efficiency](../setup/adaptation_efficiency/) | [evaluation/adaptation_efficiency](evaluation/adaptation_efficiency/) | Result note present |
| [error_audit](../setup/error_audit/) | [evaluation/error_audit](evaluation/error_audit/) | Result note present |
| [ladder_sampling_profile](../setup/ladder_sampling_profile/) | [evaluation/performance/ladder_sampling_profile](evaluation/performance/ladder_sampling_profile/) | Result note present |
| [node_regression](../setup/node_regression/) | [evaluation/task_tables/node_regression](evaluation/task_tables/node_regression/) | Analysis artifacts; completion not inferred |
| [rq1_label_efficiency_loto](../setup/rq1_label_efficiency_loto/) | [evaluation/rq1_label_efficiency_loto](evaluation/rq1_label_efficiency_loto/) | Analysis artifacts; completion not inferred |
| [static_link_prediction](../setup/static_link_prediction/) | [evaluation/task_tables/static_link_prediction](evaluation/task_tables/static_link_prediction/) | Analysis artifacts; completion not inferred |

#### Setups Without An Exact-Name Analysis Match

| Setup family | Local evidence | Documentation status |
|---|---|---|
| [_misc](../setup/_misc/) | See setup files | Setup record; completion not inferred |
| [citation_readout_breadth](../setup/citation_readout_breadth/) | See setup files | Setup record; completion not inferred |
| [covid_only](../setup/covid_only/) | See setup files | Setup record; completion not inferred |
| [covid_ukr](../setup/covid_ukr/) | See setup files | Setup record; completion not inferred |
| [cp_hk_transfer_in](../setup/cp_hk_transfer_in/) | See setup files | Setup record; completion not inferred |
| [cp_hk_twitter](../setup/cp_hk_twitter/) | See setup files | Setup record; completion not inferred |
| [finalcore_cls2500_ladders](../setup/finalcore_cls2500_ladders/) | See setup files | Setup record; completion not inferred |
| [legacy_cross_dataset_eval](../setup/legacy_cross_dataset_eval/) | See setup files | Setup record; completion not inferred |
| [mix_slp_ablation](../setup/mix_slp_ablation/) | See setup files | Setup record; completion not inferred |
| [multitask_ssl_pairs](../setup/multitask_ssl_pairs/) | See setup files | Setup record; completion not inferred |
| [multitask_ssl_rotation](../setup/multitask_ssl_rotation/) | See setup files | Setup record; completion not inferred |
| [native_model_result_matrix_overnight](../setup/native_model_result_matrix_overnight/) | See setup files | Setup record; completion not inferred |
| [nm_complete_input_audit](../setup/nm_complete_input_audit/) | See setup files | Setup record; completion not inferred |
| [nm_error_audit_split](../setup/nm_error_audit_split/) | See setup files | Setup record; completion not inferred |
| [nm_hk_mechanism](../setup/nm_hk_mechanism/) | See setup files | Setup record; completion not inferred |
| [nm_hk_support_extremes](../setup/nm_hk_support_extremes/) | See setup files | Setup record; completion not inferred |
| [nm_ladder_fillin](../setup/nm_ladder_fillin/) | See setup files | Setup record; completion not inferred |
| [nm_ladder_order_robustness-jul_23](../setup/nm_ladder_order_robustness-jul_23/) | See setup files | Setup record; completion not inferred |
| [nm_ladder_unconfined_nhop2](../setup/nm_ladder_unconfined_nhop2/) | See setup files | Setup record; completion not inferred |
| [nm_leave_one_out_finalcore](../setup/nm_leave_one_out_finalcore/) | See setup files | Setup record; completion not inferred |
| [nm_loo_schedule_signal](../setup/nm_loo_schedule_signal/) | [RESULTS.md](../setup/nm_loo_schedule_signal/RESULTS.md) | Result note present |
| [nm_mixture_diversity_heldout_cls_2k](../setup/nm_mixture_diversity_heldout_cls_2k/) | See setup files | Setup record; completion not inferred |
| [nm_model_size_eval](../setup/nm_model_size_eval/) | See setup files | Setup record; completion not inferred |
| [nm_pairwise_finalcore](../setup/nm_pairwise_finalcore/) | See setup files | Setup record; completion not inferred |
| [nm_source_stage_audit](../setup/nm_source_stage_audit/) | See setup files | Setup record; completion not inferred |
| [nm_support_geometry](../setup/nm_support_geometry/) | See setup files | Setup record; completion not inferred |
| [nm_support_resampling](../setup/nm_support_resampling/) | See setup files | Setup record; completion not inferred |
| [nm_two_source_schedule_pilot](../setup/nm_two_source_schedule_pilot/) | See setup files | Setup record; completion not inferred |
| [paper_three_seed](../setup/paper_three_seed/) | See setup files | Setup record; completion not inferred |
| [pretrain_saturation_dense](../setup/pretrain_saturation_dense/) | See setup files | Setup record; completion not inferred |
| [pretrain_saturation_existing](../setup/pretrain_saturation_existing/) | See setup files | Setup record; completion not inferred |
| [prodigy_source_lattice_lp](../setup/prodigy_source_lattice_lp/) | See setup files | Setup record; completion not inferred |
| [regression_probe_repair](../setup/regression_probe_repair/) | See setup files | Setup record; completion not inferred |
| [rq1_native_icl](../setup/rq1_native_icl/) | See setup files | Setup record; completion not inferred |
| [sampling_improvements](../setup/sampling_improvements/) | See setup files | Setup record; completion not inferred |
| [slp_evaluator_repair](../setup/slp_evaluator_repair/) | See setup files | Setup record; completion not inferred |
| [strong_mlp](../setup/strong_mlp/) | See setup files | Setup record; completion not inferred |
| [tag_citation_graphs](../setup/tag_citation_graphs/) | See setup files | Setup record; completion not inferred |
| [task_transfer](../setup/task_transfer/) | See setup files | Setup record; completion not inferred |
| [train1](../setup/train1/) | See setup files | Setup record; completion not inferred |
| [train2](../setup/train2/) | See setup files | Setup record; completion not inferred |
| [train3](../setup/train3/) | See setup files | Setup record; completion not inferred |
| [twibot20_transfer](../setup/twibot20_transfer/) | See setup files | Setup record; completion not inferred |
| [ukr_only](../setup/ukr_only/) | See setup files | Setup record; completion not inferred |
| [vision_all9_finalcore](../setup/vision_all9_finalcore/) | See setup files | Setup record; completion not inferred |
| [vision_native_cross_ssl](../setup/vision_native_cross_ssl/) | See setup files | Setup record; completion not inferred |
| [vision_native_mixture_finalcore](../setup/vision_native_mixture_finalcore/) | See setup files | Setup record; completion not inferred |

### 8.2 Analysis families and their evidence pages

Grouped by research area, then experiment folder. Titles are source-document titles,
not endorsements of older claims. Historical versions and explicitly invalidated
records remain visible so their provenance is not lost. Some README-only families
contain completed tables without a prose verdict.

#### Transfer evidence

**[nm_all9_source_complete_batch](transfer/ablations/prodigy_nm/batch_construction/nm_all9_source_complete_batch/)** — `transfer/ablations/prodigy_nm/batch_construction/nm_all9_source_complete_batch`

- [All-nine source-complete batch-9 diagnostic](transfer/ablations/prodigy_nm/batch_construction/nm_all9_source_complete_batch/FINDINGS.md)

**[nm_all9_radius_finalcore](transfer/ablations/prodigy_nm/center_sampling/nm_all9_radius_finalcore/)** — `transfer/ablations/prodigy_nm/center_sampling/nm_all9_radius_finalcore`

- [All-nine radius-controlled neighbor matching](transfer/ablations/prodigy_nm/center_sampling/nm_all9_radius_finalcore/FINDINGS.md)

**[nm_ladder_nhop2](transfer/ablations/prodigy_nm/context_depth/nm_ladder_nhop2/)** — `transfer/ablations/prodigy_nm/context_depth/nm_ladder_nhop2`

- [NM graph ladder at 2 hops — analysis](transfer/ablations/prodigy_nm/context_depth/nm_ladder_nhop2/README.md) — index/protocol

**[nm_ladder_downstream](transfer/ablations/prodigy_nm/downstream/nm_ladder_downstream/)** — `transfer/ablations/prodigy_nm/downstream/nm_ladder_downstream`

- [NM ladder downstream — findings](transfer/ablations/prodigy_nm/downstream/nm_ladder_downstream/FINDINGS.md)

**[nm_ladder_downstream_nhop2](transfer/ablations/prodigy_nm/downstream/nm_ladder_downstream_nhop2/)** — `transfer/ablations/prodigy_nm/downstream/nm_ladder_downstream_nhop2`

- [Fair-two-hop ladder downstream findings](transfer/ablations/prodigy_nm/downstream/nm_ladder_downstream_nhop2/FINDINGS.md)
- [Fair-two-hop ladder downstream analysis](transfer/ablations/prodigy_nm/downstream/nm_ladder_downstream_nhop2/README.md) — index/protocol

**[nm_ladder_gatv2](transfer/ablations/prodigy_nm/encoder_architecture/nm_ladder_gatv2/)** — `transfer/ablations/prodigy_nm/encoder_architecture/nm_ladder_gatv2`

- [NM ladder with a GATv2 background encoder — analysis](transfer/ablations/prodigy_nm/encoder_architecture/nm_ladder_gatv2/README.md) — index/protocol

**[nm_cross_source_shortcut](transfer/ablations/prodigy_nm/episode_sampling/nm_cross_source_shortcut/)** — `transfer/ablations/prodigy_nm/episode_sampling/nm_cross_source_shortcut`

- [Results — NM cross-source-shortcut test](transfer/ablations/prodigy_nm/episode_sampling/nm_cross_source_shortcut/RESULTS.md)

**[nm_ladder_global_nhop2](transfer/ablations/prodigy_nm/episode_sampling/nm_ladder_global_nhop2/)** — `transfer/ablations/prodigy_nm/episode_sampling/nm_ladder_global_nhop2`

- [Global-merged versus interleaved NM ladder figures](transfer/ablations/prodigy_nm/episode_sampling/nm_ladder_global_nhop2/README.md) — index/protocol

**[sampling_strat_comparison](transfer/ablations/prodigy_nm/episode_sampling/sampling_strat_comparison/)** — `transfer/ablations/prodigy_nm/episode_sampling/sampling_strat_comparison`

- Notebooks: [sampling_strat_comp.ipynb](transfer/ablations/prodigy_nm/episode_sampling/sampling_strat_comparison/sampling_strat_comp.ipynb). Embedded results have not been independently revalidated here.

**[nm_interventions_overnight](transfer/ablations/prodigy_nm/nm_interventions_overnight/)** — `transfer/ablations/prodigy_nm/nm_interventions_overnight`

- [Source-held-out NM intervention campaign](transfer/ablations/prodigy_nm/nm_interventions_overnight/FINDINGS.md)
- [Multi-graph NM intervention results — 2026-09-04](transfer/ablations/prodigy_nm/nm_interventions_overnight/RESULTS.md)
- [Campaign execution log](transfer/ablations/prodigy_nm/nm_interventions_overnight/RUN_LOG.md)

**[pretrain_saturation](transfer/ablations/prodigy_nm/saturation/pretrain_saturation/)** — `transfer/ablations/prodigy_nm/saturation/pretrain_saturation`

- [Pretrain saturation — findings](transfer/ablations/prodigy_nm/saturation/pretrain_saturation/FINDINGS.md)
- [Pretrain saturation — analysis](transfer/ablations/prodigy_nm/saturation/pretrain_saturation/README.md) — index/protocol

**[pretrain_saturation_nhop2](transfer/ablations/prodigy_nm/saturation/pretrain_saturation_nhop2/)** — `transfer/ablations/prodigy_nm/saturation/pretrain_saturation_nhop2`

- [Compute-matched two-hop pretrain saturation — findings](transfer/ablations/prodigy_nm/saturation/pretrain_saturation_nhop2/FINDINGS.md)
- [Compute-matched two-hop pretrain saturation — analysis](transfer/ablations/prodigy_nm/saturation/pretrain_saturation_nhop2/README.md) — index/protocol

**[labeled_mixture_diversity_cls500](transfer/ablations/prodigy_nm/source_diversity/labeled_mixture_diversity_cls500/)** — `transfer/ablations/prodigy_nm/source_diversity/labeled_mixture_diversity_cls500`

- [Labeled-mixture diversity at fixed compute and along training](transfer/ablations/prodigy_nm/source_diversity/labeled_mixture_diversity_cls500/README.md) — index/protocol
- [Results](transfer/ablations/prodigy_nm/source_diversity/labeled_mixture_diversity_cls500/RESULTS.md)

**[nm_ladder_fixed_exposure_nhop2](transfer/ablations/prodigy_nm/source_exposure/nm_ladder_fixed_exposure_nhop2/)** — `transfer/ablations/prodigy_nm/source_exposure/nm_ladder_fixed_exposure_nhop2`

- [Fixed-exposure two-hop NM ladder — findings](transfer/ablations/prodigy_nm/source_exposure/nm_ladder_fixed_exposure_nhop2/FINDINGS.md)
- [Fixed-exposure two-hop NM ladder analysis](transfer/ablations/prodigy_nm/source_exposure/nm_ladder_fixed_exposure_nhop2/README.md) — index/protocol

**[nm_ladder_sequential_nhop2](transfer/ablations/prodigy_nm/source_schedule/nm_ladder_sequential_nhop2/)** — `transfer/ablations/prodigy_nm/source_schedule/nm_ladder_sequential_nhop2`

- [Sequential two-hop ladder analysis](transfer/ablations/prodigy_nm/source_schedule/nm_ladder_sequential_nhop2/README.md) — index/protocol

**[nm_ladder_train_test_nhop2](transfer/ablations/prodigy_nm/split_integrity/nm_ladder_train_test_nhop2/)** — `transfer/ablations/prodigy_nm/split_integrity/nm_ladder_train_test_nhop2`

- [Split-aware two-hop NM ladder analysis](transfer/ablations/prodigy_nm/split_integrity/nm_ladder_train_test_nhop2/README.md) — index/protocol

**[samgpt_c5_saturation](transfer/ablations/samgpt_graphcl/saturation/samgpt_c5_saturation/)** — `transfer/ablations/samgpt_graphcl/saturation/samgpt_c5_saturation`

- [SAMGPT five-source convergence findings](transfer/ablations/samgpt_graphcl/saturation/samgpt_c5_saturation/FINDINGS.md)

**[samgpt_covid_correctness_ablation](transfer/ablations/samgpt_graphcl/saturation/samgpt_covid_correctness_ablation/)** — `transfer/ablations/samgpt_graphcl/saturation/samgpt_covid_correctness_ablation`

- [SAMGPT COVID correctness-ablation findings](transfer/ablations/samgpt_graphcl/saturation/samgpt_covid_correctness_ablation/FINDINGS.md)

**[samgpt_covid_saturation](transfer/ablations/samgpt_graphcl/saturation/samgpt_covid_saturation/)** — `transfer/ablations/samgpt_graphcl/saturation/samgpt_covid_saturation`

- [SAMGPT sampled-COVID saturation findings](transfer/ablations/samgpt_graphcl/saturation/samgpt_covid_saturation/FINDINGS.md)

**[nm_ladder](transfer/ladders/prodigy_nm/baseline/nm_ladder/)** — `transfer/ladders/prodigy_nm/baseline/nm_ladder`

- [NM interpolation ladder — complete 8-rung results](transfer/ladders/prodigy_nm/baseline/nm_ladder/RESULTS.md)

**[ladder_cross_task_eval](transfer/ladders/prodigy_nm/cross_task/ladder_cross_task_eval/)** — `transfer/ladders/prodigy_nm/cross_task/ladder_cross_task_eval`

- [Findings: source composition depends on training horizon and evaluation task](transfer/ladders/prodigy_nm/cross_task/ladder_cross_task_eval/FINDINGS.md)
- [PRODIGY ladder cross-task evaluation](transfer/ladders/prodigy_nm/cross_task/ladder_cross_task_eval/README.md) — index/protocol

**[nm_ladder_facebook](transfer/ladders/prodigy_nm/robustness/nm_ladder_facebook/)** — `transfer/ladders/prodigy_nm/robustness/nm_ladder_facebook`

- [NM ladder with Facebook inserted at rung 6 (Order D)](transfer/ladders/prodigy_nm/robustness/nm_ladder_facebook/FINDINGS.md)

**[nm_ladder_order_robustness](transfer/ladders/prodigy_nm/robustness/nm_ladder_order_robustness/)** — `transfer/ladders/prodigy_nm/robustness/nm_ladder_order_robustness`

- [NM ladder order robustness — mixture synergy](transfer/ladders/prodigy_nm/robustness/nm_ladder_order_robustness/FINDINGS.md)
- [SAMGPT native-objective ladder and specialist matrix](transfer/ladders/prodigy_nm/robustness/nm_ladder_order_robustness/FINDINGS_GRAPHCL_NATIVE.md)
- Notebooks: [analysis.ipynb](transfer/ladders/prodigy_nm/robustness/nm_ladder_order_robustness/analysis.ipynb). Embedded results have not been independently revalidated here.

**[samgpt_graphcl_ladder](transfer/ladders/samgpt_graphcl/baseline/samgpt_graphcl_ladder/)** — `transfer/ladders/samgpt_graphcl/baseline/samgpt_graphcl_ladder`

- [SAMGPT native-GraphCL ladder findings](transfer/ladders/samgpt_graphcl/baseline/samgpt_graphcl_ladder/FINDINGS.md)

**[samgpt_weak_to_strong](transfer/ladders/samgpt_graphcl/weak_to_strong/samgpt_weak_to_strong/)** — `transfer/ladders/samgpt_graphcl/weak_to_strong/samgpt_weak_to_strong`

- [SAMGPT weak-to-strong source-mixture ladder](transfer/ladders/samgpt_graphcl/weak_to_strong/samgpt_weak_to_strong/FINDINGS.md)

**[icl_arch_matrix](transfer/matrices/cross_architecture/icl_arch_matrix/)** — `transfer/matrices/cross_architecture/icl_arch_matrix`

- [Findings: one-seed, 100-update architecture matrix](transfer/matrices/cross_architecture/icl_arch_matrix/FINDINGS.md)
- [PRODIGY × VISION × GILT common-classification analysis](transfer/matrices/cross_architecture/icl_arch_matrix/README.md) — index/protocol

**[final_core](transfer/matrices/cross_model/final_core/)** — `transfer/matrices/cross_model/final_core`

- [Final native-pretext matrix and ladder](transfer/matrices/cross_model/final_core/FINDINGS.md)
- [What explains mixture performance?](transfer/matrices/cross_model/final_core/MIXTURE_EXPLANATIONS.md)

**[source_lattice_comparison](transfer/matrices/cross_model/source_lattice_comparison/)** — `transfer/matrices/cross_model/source_lattice_comparison`

- [Recent source-pair impact across models](transfer/matrices/cross_model/source_lattice_comparison/README.md) — index/protocol

**[mt_transfer_pilot](transfer/matrices/prodigy_mt/mt_transfer_pilot/)** — `transfer/matrices/prodigy_mt/mt_transfer_pilot`

- [MT transfer pilot analysis](transfer/matrices/prodigy_mt/mt_transfer_pilot/README.md) — index/protocol

**[nm_single_source_downstream](transfer/matrices/prodigy_nm/downstream/nm_single_source_downstream/)** — `transfer/matrices/prodigy_nm/downstream/nm_single_source_downstream`

- [NM single-source downstream transfer — findings](transfer/matrices/prodigy_nm/downstream/nm_single_source_downstream/FINDINGS.md)

**[entity_disjoint_eval](transfer/matrices/prodigy_nm/identity_disjoint/entity_disjoint_eval/)** — `transfer/matrices/prodigy_nm/identity_disjoint/entity_disjoint_eval`

- [Findings: exact-ID-disjoint transfer](transfer/matrices/prodigy_nm/identity_disjoint/entity_disjoint_eval/FINDINGS.md)
- [Entity-disjoint evaluation](transfer/matrices/prodigy_nm/identity_disjoint/entity_disjoint_eval/README.md) — index/protocol

**[nm_covid_midterm](transfer/matrices/prodigy_nm/merged_vs_single/nm_covid_midterm/)** — `transfer/matrices/prodigy_nm/merged_vs_single/nm_covid_midterm`

- [Results — NM covid/midterm validation](transfer/matrices/prodigy_nm/merged_vs_single/nm_covid_midterm/RESULTS.md)

**[nm_transfer_matrix](transfer/matrices/prodigy_nm/merged_vs_single/nm_transfer_matrix/)** — `transfer/matrices/prodigy_nm/merged_vs_single/nm_transfer_matrix`

- [Results — NM transfer matrix (fair single-vs-merged)](transfer/matrices/prodigy_nm/merged_vs_single/nm_transfer_matrix/RESULTS.md)

**[nm_single_source_matrix](transfer/matrices/prodigy_nm/single_source/nm_single_source_matrix/)** — `transfer/matrices/prodigy_nm/single_source/nm_single_source_matrix`

- [NM single-source transfer matrix — findings](transfer/matrices/prodigy_nm/single_source/nm_single_source_matrix/FINDINGS.md)

**[nm_single_source_matrix_facebook](transfer/matrices/prodigy_nm/single_source/nm_single_source_matrix_facebook/)** — `transfer/matrices/prodigy_nm/single_source/nm_single_source_matrix_facebook`

- [NM single-source matrix with Facebook — findings](transfer/matrices/prodigy_nm/single_source/nm_single_source_matrix_facebook/FINDINGS.md)

**[social_specificity_pilot](transfer/matrices/prodigy_nm/single_source/social_specificity_pilot/)** — `transfer/matrices/prodigy_nm/single_source/social_specificity_pilot`

- [Social-specificity transfer pilot](transfer/matrices/prodigy_nm/single_source/social_specificity_pilot/FINDINGS.md)

**[nm_pairwise_source_additions](transfer/matrices/prodigy_nm/source_additions/nm_pairwise_source_additions/)** — `transfer/matrices/prodigy_nm/source_additions/nm_pairwise_source_additions`

- [NM pairwise source additions](transfer/matrices/prodigy_nm/source_additions/nm_pairwise_source_additions/README.md) — index/protocol

#### Objectives evidence

**[pretrain_probe_matrix](objectives/frozen_probes/pretrain_probe_matrix/)** — `objectives/frozen_probes/pretrain_probe_matrix`

- [Pretraining-Strategy Probe Matrix — Findings](objectives/frozen_probes/pretrain_probe_matrix/FINDINGS.md)
- [pretrain_probe_matrix — Findings](objectives/frozen_probes/pretrain_probe_matrix/FINDINGS_v1_archived.md) — historical version

**[best_pretrain_strat](objectives/legacy/best_pretrain_strat/)** — `objectives/legacy/best_pretrain_strat`

- Notebooks: [best.ipynb](objectives/legacy/best_pretrain_strat/best.ipynb), [strat_comparison.ipynb](objectives/legacy/best_pretrain_strat/strat_comparison.ipynb), [strat_comparison_1.ipynb](objectives/legacy/best_pretrain_strat/strat_comparison_1.ipynb). Embedded results have not been independently revalidated here.

**[pretrain_strategy_benchmark](objectives/legacy/pretrain_strategy_benchmark/)** — `objectives/legacy/pretrain_strategy_benchmark`

- Notebooks: [pretrain_strategy_benchmark.ipynb](objectives/legacy/pretrain_strategy_benchmark/pretrain_strategy_benchmark.ipynb). Embedded results have not been independently revalidated here.

**[multitask_ssl](objectives/multitask/multitask_ssl/)** — `objectives/multitask/multitask_ssl`

- [Findings — the mixed-objective lattice on valid metrics only](objectives/multitask/multitask_ssl/FINDINGS.md)
- [Static link prediction, rescored on a valid evaluator](objectives/multitask/multitask_ssl/FINDINGS_rescore.md)
- [Multitask SSL — the mixed-objective experiments](objectives/multitask/multitask_ssl/README.md) — index/protocol

**[multitask_ssl_corpora](objectives/multitask/multitask_ssl_corpora/)** — `objectives/multitask/multitask_ssl_corpora`

- [multitask_ssl_corpora — findings](objectives/multitask/multitask_ssl_corpora/FINDINGS.md)

**[feature_ablation](objectives/topology_vs_features/feature_ablation/)** — `objectives/topology_vs_features/feature_ablation`

- [feature_ablation — Findings](objectives/topology_vs_features/feature_ablation/FINDINGS.md)

**[topology_feature_ssl](objectives/topology_vs_features/topology_feature_ssl/)** — `objectives/topology_vs_features/topology_feature_ssl`

- [topology_feature_ssl — Findings](objectives/topology_vs_features/topology_feature_ssl/FINDINGS.md)
- [topology_feature_ssl — Findings (B0 / B1 / E1 / E2 / E2b vs trivial floor)](objectives/topology_vs_features/topology_feature_ssl/FINDINGS_v1_archived.md) — historical version
- [topology_feature_ssl — RESULTS](objectives/topology_vs_features/topology_feature_ssl/RESULTS.md)
- [directed3_log (input-scaling fix) vs original — topology_feature_ssl](objectives/topology_vs_features/topology_feature_ssl/RESULTS_directed3log.md)
- [topology_feature_ssl — matched-40k results (B0/B1/E1/E2/E2b vs trivial floor)](objectives/topology_vs_features/topology_feature_ssl/RESULTS_matched40k.md)
- Notebooks: [analysis.ipynb](objectives/topology_vs_features/topology_feature_ssl/analysis.ipynb), [results.ipynb](objectives/topology_vs_features/topology_feature_ssl/results.ipynb), [topology_feature_ssl.ipynb](objectives/topology_vs_features/topology_feature_ssl/topology_feature_ssl.ipynb). Embedded results have not been independently revalidated here.

**[covid_task_transfer_matrix](objectives/within_dataset_transfer/covid_task_transfer_matrix/)** — `objectives/within_dataset_transfer/covid_task_transfer_matrix`

- Notebooks: [covid_task_transfer_matrix.ipynb](objectives/within_dataset_transfer/covid_task_transfer_matrix/covid_task_transfer_matrix.ipynb). Embedded results have not been independently revalidated here.

#### Graphs evidence

**[nm_pinsage_fast_ablation](graphs/encoders/nm_pinsage_fast_ablation/)** — `graphs/encoders/nm_pinsage_fast_ablation`

- [PinSAGE fixed-test result](graphs/encoders/nm_pinsage_fast_ablation/FINDINGS.md)

**[bio_embedding_geometry](graphs/features/bio_embedding_geometry/)** — `graphs/features/bio_embedding_geometry`

- [Findings: geometry of bio embeddings across nine graphs](graphs/features/bio_embedding_geometry/FINDINGS.md)
- [Bio-embedding geometry across the nine social graphs](graphs/features/bio_embedding_geometry/README.md) — index/protocol

**[identity_overlap_audit](graphs/overlap/identity_overlap_audit/)** — `graphs/overlap/identity_overlap_audit`

- [Findings: cross-dataset identity-overlap audit](graphs/overlap/identity_overlap_audit/FINDINGS.md)
- [Identity-overlap audit analysis](graphs/overlap/identity_overlap_audit/README.md) — index/protocol

**[graph_divergence](graphs/structure/graph_divergence/)** — `graphs/structure/graph_divergence`

- Notebooks: [graph_divergence.ipynb](graphs/structure/graph_divergence/graph_divergence.ipynb). Embedded results have not been independently revalidated here.

**[path_feature_coupling](graphs/structure_features/path_feature_coupling/)** — `graphs/structure_features/path_feature_coupling`

- [Findings: path length versus node-feature distance](graphs/structure_features/path_feature_coupling/FINDINGS.md)
- [Path length versus node-feature distance](graphs/structure_features/path_feature_coupling/README.md) — index/protocol

**[.pytest_cache](graphs/structure_features/path_feature_coupling/.pytest_cache/)** — `graphs/structure_features/path_feature_coupling/.pytest_cache`

- [pytest cache directory #](graphs/structure_features/path_feature_coupling/.pytest_cache/README.md) — index/protocol

**[centered_ridge_training](graphs/transfer_prediction/centered_ridge_training/)** — `graphs/transfer_prediction/centered_ridge_training`

- [Strengthened direct training closes the representation-utility gap](graphs/transfer_prediction/centered_ridge_training/FINDINGS.md)

**[encoder_solver_isolation](graphs/transfer_prediction/encoder_solver_isolation/)** — `graphs/transfer_prediction/encoder_solver_isolation`

- [Encoder–solver isolation: completed pilot, AUC no-go](graphs/transfer_prediction/encoder_solver_isolation/FINDINGS.md)

**[gilt_component_crossover](graphs/transfer_prediction/gilt_component_crossover/)** — `graphs/transfer_prediction/gilt_component_crossover`

- [GILT scope test: compensation pattern did not replicate](graphs/transfer_prediction/gilt_component_crossover/FINDINGS.md)

**[local_transfer_contrast](graphs/transfer_prediction/local_transfer_contrast/)** — `graphs/transfer_prediction/local_transfer_contrast`

- [Local transfer contrast: source-induced decision deformation](graphs/transfer_prediction/local_transfer_contrast/FINDINGS.md)

**[public_prodigy_kg](graphs/transfer_prediction/public_prodigy_kg/)** — `graphs/transfer_prediction/public_prodigy_kg`

- [Public KG support/query context experiment](graphs/transfer_prediction/public_prodigy_kg/FINDINGS.md)

**[similarity_vs_transfer](graphs/transfer_prediction/similarity_vs_transfer/)** — `graphs/transfer_prediction/similarity_vs_transfer`

- [Findings: does graph similarity predict single-source NM transfer?](graphs/transfer_prediction/similarity_vs_transfer/FINDINGS.md)
- Notebooks: [similarity_vs_transfer.ipynb](graphs/transfer_prediction/similarity_vs_transfer/similarity_vs_transfer.ipynb). Embedded results have not been independently revalidated here.

**[similarity_vs_transfer_v2](graphs/transfer_prediction/similarity_vs_transfer_v2/)** — `graphs/transfer_prediction/similarity_vs_transfer_v2`

- [What predicts GNN transfer? Nine-graph predictor study (v2)](graphs/transfer_prediction/similarity_vs_transfer_v2/FINDINGS.md)

**[support_boundary_repair](graphs/transfer_prediction/support_boundary_repair/)** — `graphs/transfer_prediction/support_boundary_repair`

- [Support-only boundary repair: close the tested deployment direction](graphs/transfer_prediction/support_boundary_repair/FINDINGS.md)
- [Return to the pretraining question](graphs/transfer_prediction/support_boundary_repair/NEXT_DECISION.md)

**[target_performance_mechanisms](graphs/transfer_prediction/target_performance_mechanisms/)** — `graphs/transfer_prediction/target_performance_mechanisms`

- [When training stops exploiting what the target rewards](graphs/transfer_prediction/target_performance_mechanisms/ARGUMENT_CUE_DEPENDENT_TRANSFER.md)
- [Separate what transfers from how it is used](graphs/transfer_prediction/target_performance_mechanisms/ARGUMENT_CURRENT_DECISION.md)
- [A flat transfer score can conceal opposing component changes](graphs/transfer_prediction/target_performance_mechanisms/ARGUMENT_READOUT_DEPENDENT_TRANSFER.md)
- [Transfer scores hide opposing changes in graph-ICL components](graphs/transfer_prediction/target_performance_mechanisms/ARGUMENT_REPLICATION_DECISION.md)
- [What would readers understand or do differently?](graphs/transfer_prediction/target_performance_mechanisms/CONTRIBUTION_REQUIREMENTS.md)
- [One independently nominated natural-pair K/V prediction](graphs/transfer_prediction/target_performance_mechanisms/DECISION_KV_GENERALITY.md)
- [Advisor decision: do not restart a support-geometry selector](graphs/transfer_prediction/target_performance_mechanisms/DECISION_PREDICTOR_AUDIT.md)
- [Next decisive question: task-selective transfer at fixed graph inputs](graphs/transfer_prediction/target_performance_mechanisms/DECISION_TASK_SWITCH.md)
- [Fixed temporal role-bridge decision](graphs/transfer_prediction/target_performance_mechanisms/DECISION_TEMPORAL_ROLE_BRIDGE.md)
- [Target performance: what the completed mixture lattice establishes](graphs/transfer_prediction/target_performance_mechanisms/FINDINGS.md)
- [Support-side repair is not confined to explicit-cue queries](graphs/transfer_prediction/target_performance_mechanisms/FINDINGS_ANNOTATION_CUES.md)
- [Contrast compression with a surviving class preference](graphs/transfer_prediction/target_performance_mechanisms/FINDINGS_CLASS_REFERENCE_ACCOUNTING.md)
- [Decision: graph context constructs a learned classifier](graphs/transfer_prediction/target_performance_mechanisms/FINDINGS_CLASS_REFERENCE_DECISION.md)
- [Decision: support values, not routing alone, carry the political ranking effect](graphs/transfer_prediction/target_performance_mechanisms/FINDINGS_CLASS_REFERENCE_KV.md)
- [The late suppressed supports collapse before the metagraph](graphs/transfer_prediction/target_performance_mechanisms/FINDINGS_COLLAPSE_LOCATION.md)
- [A shared support shift is not sufficient for the 50k ranking gain](graphs/transfer_prediction/target_performance_mechanisms/FINDINGS_COMMON_SUPPORT_SHIFT.md)
- [Actual training examples: identity conflicts depend on support/query roles](graphs/transfer_prediction/target_performance_mechanisms/FINDINGS_CONSUMED_CONFLICTS.md)
- [Contribution decision after the public boundary and related-work check](graphs/transfer_prediction/target_performance_mechanisms/FINDINGS_CONTRIBUTION_POSITIONING.md)
- [Source usefulness changes across trained families and readouts](graphs/transfer_prediction/target_performance_mechanisms/FINDINGS_CROSS_MODEL.md)
- [Early support benefit is not concentrated in one dominant value contribution](graphs/transfer_prediction/target_performance_mechanisms/FINDINGS_EARLY_SUPPORT_CONTRIBUTIONS.md)
- [Individual successes and failures: first matched case review](graphs/transfer_prediction/target_performance_mechanisms/FINDINGS_EXAMPLES.md)
- [Resampling the same supports is not the political-transfer repair](graphs/transfer_prediction/target_performance_mechanisms/FINDINGS_FIXED_SUPPORT_CONTEXT.md)
- [Nominated contrast: reference collapse and initialization sensitivity](graphs/transfer_prediction/target_performance_mechanisms/FINDINGS_INITIALIZATION_PREDICTION.md)
- [Claim audit: class-reference transfer after the nominated pairing](graphs/transfer_prediction/target_performance_mechanisms/FINDINGS_KV_CLAIM_AUDIT.md)
- [Nominated test: label-initialization by support-context interaction](graphs/transfer_prediction/target_performance_mechanisms/FINDINGS_LABEL_INTERFACE_DECISION.md)
- [Matched page trajectory: catch-up, not absolute deterioration](graphs/transfer_prediction/target_performance_mechanisms/FINDINGS_MATCHED_PAGE_CROSSOVER.md)
- [Background-message presence matters more than the tested neighbor content](graphs/transfer_prediction/target_performance_mechanisms/FINDINGS_MESSAGE_CONTENT.md)
- [Message-scale decision: support direction, not a pure norm effect](graphs/transfer_prediction/target_performance_mechanisms/FINDINGS_MESSAGE_SCALE_DECISION.md)
- [Valid support examples change which fixed queries a model gets right](graphs/transfer_prediction/target_performance_mechanisms/FINDINGS_NATURAL_SUPPORT.md)
- [Same seed and inputs do not guarantee the same trained model](graphs/transfer_prediction/target_performance_mechanisms/FINDINGS_NUMERICS.md)
- [Offset subtraction locates an ablation effect, not a transfer repair](graphs/transfer_prediction/target_performance_mechanisms/FINDINGS_OFFSET_INTERVENTION.md)
- [Paper synthesis: distinguish reference repair from fragile ranking gains](graphs/transfer_prediction/target_performance_mechanisms/FINDINGS_PAPER_SYNTHESIS.md)
- [Public boundary: useful context in both roles, with cross-role coupling](graphs/transfer_prediction/target_performance_mechanisms/FINDINGS_PUBLIC_BOUNDARY.md)
- [Public temporal crossover: compatibility, not a deteriorating inference module](graphs/transfer_prediction/target_performance_mechanisms/FINDINGS_PUBLIC_CROSSOVER_BRIDGE.md)
- [Repeated-query contexts do not resolve the source-performance explanation](graphs/transfer_prediction/target_performance_mechanisms/FINDINGS_QUERY_EXCHANGEABILITY.md)
- [Twenty supports can choose a better cross-target prediction rule](graphs/transfer_prediction/target_performance_mechanisms/FINDINGS_READOUT_SELECTION.md)
- [Fixed-routing reference transport is affine, not an unidentified nonlinearity](graphs/transfer_prediction/target_performance_mechanisms/FINDINGS_REFERENCE_TRANSPORT.md)
- [Support-calibrated repair exchanges errors rather than preserving ridge decisions](graphs/transfer_prediction/target_performance_mechanisms/FINDINGS_REPAIR_ERROR_ACCOUNTING.md)
- [Mechanism results: target signal, learned readout, and a failed coverage prediction](graphs/transfer_prediction/target_performance_mechanisms/FINDINGS_REPLAY.md)
- [Query context versus support context: from cases to complete replay](graphs/transfer_prediction/target_performance_mechanisms/FINDINGS_ROLE_CONTEXT.md)
- [Role interaction: correction to the manuscript's removal narrative](graphs/transfer_prediction/target_performance_mechanisms/FINDINGS_ROLE_INTERACTION.md)
- [Degree-preserving rewiring and a second-target support repair](graphs/transfer_prediction/target_performance_mechanisms/FINDINGS_ROLE_TOPOLOGY.md)
- [Schedule effects depend on the readout](graphs/transfer_prediction/target_performance_mechanisms/FINDINGS_SCHEDULE_READOUT.md)
- [Source choice depends on readout, but decision benefit is limited](graphs/transfer_prediction/target_performance_mechanisms/FINDINGS_SOURCE_CHOICE_CONSEQUENCE.md)
- [Support suppression has opposite dose-response curves across targets](graphs/transfer_prediction/target_performance_mechanisms/FINDINGS_SUPPORT_DOSE.md)
- [Matched support-label binding: a learned, source-dependent effect](graphs/transfer_prediction/target_performance_mechanisms/FINDINGS_SUPPORT_GRADIENTS.md)
- [Suppression does not uniformly improve support geometry](graphs/transfer_prediction/target_performance_mechanisms/FINDINGS_SUPPORT_PROTOTYPE_CONTRAST.md)
- [Class weighting does not explain the support-validation reversal](graphs/transfer_prediction/target_performance_mechanisms/FINDINGS_SUPPORT_WEIGHTING.md)
- [Temporal crossover: closest prior and the contribution boundary](graphs/transfer_prediction/target_performance_mechanisms/FINDINGS_TEMPORAL_NOVELTY_BOUNDARY.md)
- [Trajectory contribution decision — 7 September 2026](graphs/transfer_prediction/target_performance_mechanisms/FINDINGS_TRAJECTORY_DECISION.md)
- [Zero messages expose a large learned affine offset](graphs/transfer_prediction/target_performance_mechanisms/FINDINGS_ZERO_MESSAGE_OFFSET.md)
- [Fresh-chat handoff — 7 September 2026](graphs/transfer_prediction/target_performance_mechanisms/HANDOFF_FRESH_CHAT_20260907.md)
- [Portable inference reproduction: complete final-checkpoint replay](graphs/transfer_prediction/target_performance_mechanisms/PORTABLE_REPRODUCTION.md)
- [Private artifact inventory](graphs/transfer_prediction/target_performance_mechanisms/PRIVATE_ARTIFACTS.md)
- [Target-performance mechanisms](graphs/transfer_prediction/target_performance_mechanisms/README.md) — index/protocol

**[trace_health_guided](graphs/transfer_prediction/trace_health_guided/)** — `graphs/transfer_prediction/trace_health_guided`

- [Health-guided source allocation does not improve the primary comparison](graphs/transfer_prediction/trace_health_guided/FINDINGS.md)
- [Value replacement improves ranking but collapses classification](graphs/transfer_prediction/trace_health_guided/FINDINGS_VALUE_OPERATING_POINT.md)
- [Research synthesis: support context constructs the classifier](graphs/transfer_prediction/trace_health_guided/MECHANISM_SYNTHESIS.md)
- [Public replication: decision before target outcomes](graphs/transfer_prediction/trace_health_guided/PUBLIC_REPLICATION_DECISION.md)
- [Contribution audit against nearby work](graphs/transfer_prediction/trace_health_guided/RELATED_WORK_AUDIT.md)

**[trace_schedule_scaling](graphs/transfer_prediction/trace_schedule_scaling/)** — `graphs/transfer_prediction/trace_schedule_scaling`

- [TRACE schedule scaling — findings](graphs/transfer_prediction/trace_schedule_scaling/FINDINGS.md)
- [Schedule effects depend on the readout](graphs/transfer_prediction/trace_schedule_scaling/FINDINGS_STAGE_AUDIT.md)
- [TRACE schedule scaling analysis](graphs/transfer_prediction/trace_schedule_scaling/README.md) — index/protocol

#### Evaluation evidence

**[adaptation_efficiency](evaluation/adaptation_efficiency/)** — `evaluation/adaptation_efficiency`

- [Adaptation-efficiency results](evaluation/adaptation_efficiency/FINDINGS.md)
- [Frozen-encoder adaptation-efficiency analysis](evaluation/adaptation_efficiency/README.md) — index/protocol

**[error_audit](evaluation/error_audit/)** — `evaluation/error_audit`

- [Source-paired COVID Political error audit](evaluation/error_audit/FINDINGS_COVID_POLITICAL_SOURCE_PAIR.md)
- [NM bio clusters, repeated queries, and anchor ambiguity](evaluation/error_audit/FINDINGS_NM_BIO_CLUSTERS.md)
- [Canonical-split NM error audit](evaluation/error_audit/FINDINGS_NM_CANONICAL_SPLIT.md)
- [Canonical NM: complete sampled-input failure analysis](evaluation/error_audit/FINDINGS_NM_COMPLETE_INPUTS.md)
- [Corrected NM degree audit](evaluation/error_audit/FINDINGS_NM_DEGREE.md)
- [Canonical NM: query, support, and episode difficulty](evaluation/error_audit/FINDINGS_NM_DIFFICULTY.md)
- [Canonical NM: what distinguishes failures from successes?](evaluation/error_audit/FINDINGS_NM_EPISODE_DETAIL.md)
- [HK NM failures: membership ambiguity and class-reference construction](evaluation/error_audit/FINDINGS_NM_HK_FAILURE_LOCALIZATION.md)
- [HK→HK NM: value-path mechanism and a failed bounded repair](evaluation/error_audit/FINDINGS_NM_HK_GOAL.md)
- [HK: frozen encoder versus metagraph under support changes](evaluation/error_audit/FINDINGS_NM_HK_MECHANISM.md)
- [HK: deliberately selecting support similarity extremes](evaluation/error_audit/FINDINGS_NM_HK_SUPPORT_EXTREMES.md)
- [Source-paired neighbor-matching error audit](evaluation/error_audit/FINDINGS_NM_SOURCE_PAIR.md)
- [Canonical NM: where native and foreign models diverge](evaluation/error_audit/FINDINGS_NM_SOURCE_STAGES.md)
- [Fixed-query support changes: does input separation track the flips?](evaluation/error_audit/FINDINGS_NM_SUPPORT_GEOMETRY.md)
- [Canonical NM: paired support resampling](evaluation/error_audit/FINDINGS_NM_SUPPORT_RESAMPLING.md)
- [Query bio embedding clusters and classification errors](evaluation/error_audit/FINDINGS_QUERY_BIO_CLUSTERS.md)
- [Error-audit analysis](evaluation/error_audit/README.md) — index/protocol
- [Source-paired episode error audit](evaluation/error_audit/SOURCE_PAIR_ERROR_AUDIT_REPORT.md)

**[ledger](evaluation/ledger/)** — `evaluation/ledger`

- [Unified experiment inventory](evaluation/ledger/EXPERIMENT_INVENTORY.md)
- [Experiment taxonomy](evaluation/ledger/EXPERIMENT_TAXONOMY.md)
- [Current evaluation ledger](evaluation/ledger/README.md) — index/protocol
- [W&B experiment inventory](evaluation/ledger/WANDB_EXPERIMENT_INVENTORY.md)

**[ladder_sampling_profile](evaluation/performance/ladder_sampling_profile/)** — `evaluation/performance/ladder_sampling_profile`

- [PRODIGY ladder pipeline profile — 2026-09-03](evaluation/performance/ladder_sampling_profile/FINDINGS.md)
- [Eight-model shared-graph validation — 2026-09-04](evaluation/performance/ladder_sampling_profile/SHARED_TRAINING_VALIDATION.md)

**[rq1_label_efficiency_loto](evaluation/rq1_label_efficiency_loto/)** — `evaluation/rq1_label_efficiency_loto`

- [Literature note: reporting label efficiency in GNN and graph-foundation-model experiments](evaluation/rq1_label_efficiency_loto/LABEL_BUDGET_REPORTING.md)
- [RQ1 label-efficient unseen-family transfer](evaluation/rq1_label_efficiency_loto/README.md) — index/protocol

**[static_link_prediction_repair](evaluation/static_link_prediction_repair/)** — `evaluation/static_link_prediction_repair`

- [Static-LP evaluator repair](evaluation/static_link_prediction_repair/README.md) — index/protocol

**[node_regression](evaluation/task_tables/node_regression/)** — `evaluation/task_tables/node_regression`

- Notebooks: [node_regression.ipynb](evaluation/task_tables/node_regression/node_regression.ipynb). Embedded results have not been independently revalidated here.

**[static_link_prediction](evaluation/task_tables/static_link_prediction/)** — `evaluation/task_tables/static_link_prediction`

- Notebooks: [static_link_prediction.ipynb](evaluation/task_tables/static_link_prediction/static_link_prediction.ipynb). Embedded results have not been independently revalidated here.

#### Synthesis evidence

**[cross_experiment](synthesis/cross_experiment/)** — `synthesis/cross_experiment`

- [NM Cross-Source Transfer Study: Merged vs. Single-Source Pretraining](synthesis/cross_experiment/NM_CROSS_SOURCE_STUDY.md)
- [NM merged-vs-single: cross-experiment summary](synthesis/cross_experiment/NM_MERGED_VS_SINGLE_SUMMARY.md)
- [GFM Retweet-Graph Program — Consolidated Findings](synthesis/cross_experiment/PROGRAM_FINDINGS.md)
- [Cross-experiment syntheses](synthesis/cross_experiment/README.md) — index/protocol

**[native_model_result_matrix](synthesis/cross_experiment/native_model_result_matrix/)** — `synthesis/cross_experiment/native_model_result_matrix`

- [Figure index](synthesis/cross_experiment/native_model_result_matrix/FIGURE_INDEX.md)
- [Native-model result-matrix audit](synthesis/cross_experiment/native_model_result_matrix/FINDINGS.md)
- [Native-model result matrix](synthesis/cross_experiment/native_model_result_matrix/README.md) — index/protocol
- [Execution log](synthesis/cross_experiment/native_model_result_matrix/RUN_LOG.md)

**[new_plot_suite](synthesis/cross_experiment/new_plot_suite/)** — `synthesis/cross_experiment/new_plot_suite`

- [New plot suite](synthesis/cross_experiment/new_plot_suite/README.md) — index/protocol

**[paper_vision_evidence](synthesis/cross_experiment/paper_vision_evidence/)** — `synthesis/cross_experiment/paper_vision_evidence`

- [Downstream evidence for the multi-graph pretraining paper](synthesis/cross_experiment/paper_vision_evidence/FINDINGS.md)
- [Paper vision evidence](synthesis/cross_experiment/paper_vision_evidence/README.md) — index/protocol

### 8.3 Findings outside the main worktree

Historical September 11 worktree inventory. During the September 12 closeout, links
were redirected to preserved main-tree records or immutable commits on the remote.
The worktree paths and revisions below describe their original locations, not
currently existing checkouts. This does not certify every branch result as final.

**`codex/hk-lp-baselines-20260908`**, observed revision `ebefbf791578`; worktree `/private/tmp/prodigy-hk-lp-baselines`

- [HK direct-link baseline pilot — 2026-09-08](evaluation/lp_baselines/hk_lp_baselines/FINDINGS.md)

**`codex/nm-hk-goal-20260908`**, observed revision `9949a5e3fd34`; worktree `/private/tmp/prodigy-nm-hk-goal`

- [NM broad native/foreign stage audit](evaluation/error_audit/FINDINGS_NM_BROAD_SOURCE_STAGES.md)
- [NM failure hierarchy: from one HK query to source-dependent transfer](evaluation/error_audit/FINDINGS_NM_FAILURE_HIERARCHY.md)
- [HK→HK NM: full class-reference and native/foreign decomposition](evaluation/error_audit/FINDINGS_NM_HK_CLASS_REFERENCE_FULL.md)
- [Canonical HK NM failure decomposition](evaluation/error_audit/FINDINGS_NM_HK_FAILURE_DECOMPOSITION.md)
- [HK→HK NM: failure phenotypes inside the large residual](evaluation/error_audit/FINDINGS_NM_HK_FAILURE_PHENOTYPES.md)
- [HK→HK NM: first query and its complete episode](evaluation/error_audit/FINDINGS_NM_HK_FIRST_QUERY_EPISODE.md)
- [HK→HK NM: multi-positive evaluation and contradicted-negative masking](evaluation/error_audit/FINDINGS_NM_HK_OVERLAP_AWARE.md)
- [HK→HK NM: matched overlap-aware training intervention](evaluation/error_audit/FINDINGS_NM_HK_OVERLAP_TRAINING.md)
- [HK→HK NM: bridge from realized failures to training supervision](evaluation/error_audit/FINDINGS_NM_HK_TRAINING_CONFLICT_BRIDGE.md)
- [Native-graph replication: HK and Ukraine NM class references](evaluation/error_audit/FINDINGS_NM_NATIVE_GRAPH_REFERENCE_REPLICATION.md)
- [NM source-manifold affinity on fixed HK and Ukraine inputs](evaluation/error_audit/FINDINGS_NM_SOURCE_MANIFOLD.md)
- [Native versus foreign NM separation before and after the metagraph](evaluation/error_audit/FINDINGS_NM_SOURCE_MARGIN_STAGES.md)

**`codex/proxy-a-seeds-clean-20260909`**, observed revision `f42bc9c87eb3`; worktree `/Users/philipp/projects/gfm/prodigy-proxy-a-seeds`

- [Proxy-A: three-seed stability](https://github.com/usc-isi-i2/prodigy/blob/1eadf1d9c807f58b6fe81547c3b9f1e1b3c2ea5d/scripts/experiments/analysis/graphs/transfer_prediction/proxy_a_seed_stability/FINDINGS.md)
- [Does proxy-A explain compatibility beyond donor strength?](https://github.com/usc-isi-i2/prodigy/blob/1eadf1d9c807f58b6fe81547c3b9f1e1b3c2ea5d/scripts/experiments/analysis/graphs/transfer_prediction/proxy_a_seed_stability/incremental/FINDINGS.md)
- [Proxy-A: three-seed stability](https://github.com/usc-isi-i2/prodigy/blob/1eadf1d9c807f58b6fe81547c3b9f1e1b3c2ea5d/scripts/experiments/analysis/graphs/transfer_prediction/proxy_a_seed_stability/sample_40000/FINDINGS.md)

**`codex/social-to-fb15k-adapter`**, observed revision `1ea962b88257`; worktree `/Users/philipp/projects/gfm/prodigy-social-fb15k`

- [Social-to-FB15K-237 compatibility transfer](transfer/matrices/prodigy_nm/downstream/social_to_fb15k_adapter/FINDINGS.md)

### 8.4 Sibling-project evidence and checkout map

The named experimental families are summarized in §3.4 and §4.2. Other checkouts
below are retained for discoverability; checkout names do not establish distinct
experiments or completed results.

| Family | Local project/checkouts | Primary record |
|---|---|---|
| GraphSAGE/GraphMAE/MLP mixture studies | Preserved in `philippnoah/mixture-scaling`; former local analysis worktrees were removed after push. | [Study](https://github.com/philippnoah/mixture-scaling/blob/da521b8aec95859ab366bad8723c418273a719c5/results/STRICT_STUDY.md), [lattice](https://github.com/philippnoah/mixture-scaling/blob/da521b8aec95859ab366bad8723c418273a719c5/docs/social9_source_lattice.md), experiment-specific protocol links in §3.4 |
| SAMGPT native and downstream studies | Preserved in `philippnoah/samgpt-social`; former local analysis worktrees were removed after push. | [Order C](https://github.com/philippnoah/samgpt-social/blob/ab944447e929c34eade772c9a5ee700b6f8c611e/analysis/mixture_order_c/FINDINGS.md), [weak-to-strong](https://github.com/philippnoah/samgpt-social/blob/ab944447e929c34eade772c9a5ee700b6f8c611e/analysis/mixture_weak_to_strong/FINDINGS.md), [lattice](https://github.com/philippnoah/samgpt-social/blob/ab944447e929c34eade772c9a5ee700b6f8c611e/analysis/source_lattice/FINDINGS.md) |
| GraphGlue reproduction/pilot | Three-seed results are preserved in the private `philippnoah/graphglue-results` repository; completed one-epoch results remain here. | [Three-seed final report](https://github.com/philippnoah/graphglue-results/blob/619e1903712f28a4f5deab876567f08a80c36edb/FINAL_REPORT.md), [earlier report](https://github.com/philippnoah/graphglue-results/blob/619e1903712f28a4f5deab876567f08a80c36edb/REPORT.md); [completed one-epoch ladder](transfer/ladders/graphglue/graphglue_ladder_oneepoch/FINDINGS.md) |
| Topology controls | Preserved in `philippnoah/gin-topology`; former local checkouts were removed after push. | [GIN result](https://github.com/philippnoah/gin-topology/blob/c02f2394dec78800ec82606a68e383483fc4825e/RESULTS.md); upstream code presence is not a separate completed study |
| Proxy-A null | Complete findings, code, results and sample caches are preserved in the private `philippnoah/proxy-a-null-20260909` repository. | [Null-control findings](https://github.com/philippnoah/proxy-a-null-20260909/blob/b131a936705ef575be9907f4b7d1ac1e8a9d3e02/FINDINGS.md) |

## 9. June and earlier archived work

The preserved archive mixes May precursors with June work; names without an explicit
date are not assigned a fabricated run date. These studies are mostly notebooks with
embedded outputs. They are historical provenance, not automatically current evidence.
The controlled June/July successors above take precedence where they correct protocols.

### 9.1 Retired notebook families

The following records were enumerated from tag
`archive/retired-analyses-2026-07-26`; see the [archive guide](archive/README.md).
They remain in git and were not restored into the working tree.

| Archived family | Historical purpose / record | Current treatment |
|---|---|---|
| `aug` | Augmentation exploration; notebook has no opening prose summary. | Archived; no new validation |
| `compare_amandeep_train2` | Earlier model/run comparison; no opening prose summary. | Archived; no new validation |
| `covid_only_ukr_only` | June 15 COVID-only versus Ukraine-only across tasks and shot counts. | Archived; no new validation |
| `covid_political_pl_trajectory` | Classification trajectory on COVID Political for single/merged NM models. | Archived; no new validation |
| `covid_political_pl_trajectory_aug` | Augmented Ukraine+COVID model classification trajectory. | Archived; no new validation |
| `covid_ukr` | Cross-graph evaluation grid by dataset, task and shots. | Archived; no new validation |
| `covid_vs_ukr_vs_merged` | Single-source COVID/Ukraine versus their merge. | Archived; no new validation |
| `cp_hk_transfer_in` | Matched-checkpoint transfer into HK, 3-way and 30-way NM. | Archived; no new validation |
| `cp_hk_transfer_scatter` | HK transfer scatter artifacts. | Archived; no new validation |
| `embedding_ablation` | Embedding-ablation exploration; notebook prose is minimal. | Archived; no new validation |
| `episode_viewer` | Interactive few-shot episode inspection tooling. | Archived; no new validation |
| `eval_merged_11_06_2026` | June 11 merged-model evaluation exports. | Archived; no new validation |
| `iohunter` | Earlier IOHunter analysis; no opening prose summary. | Archived; no new validation |
| `runs_cleaned_may20.csv` | Historical exported data; inspect the git archive before interpreting. | Archived; no new validation |
| `runs_cleaned_may21.csv` | Historical exported data; inspect the git archive before interpreting. | Archived; no new validation |
| `runs_cleaned_may7.csv` | Historical exported data; inspect the git archive before interpreting. | Archived; no new validation |
| `social_llm` | Earlier social-LLM analysis; no opening prose summary. | Archived; no new validation |
| `train1` | Early COVID-to-Midterm and COVID/Ukraine transfer plots. | Archived; no new validation |
| `train2` | Earlier training/evaluation analysis; notebook prose is minimal. | Archived; no new validation |
| `train3` | Early full transfer matrix. | Archived; no new validation |
| `transfer_trajectory_merged_ukr_rus_covid_midterm_nm_aug` | Augmented three-source transfer trajectories at 1k/10k/50k/100k. | Archived; no new validation |
| `twibot20_transfer_in` | Merged-strategy checkpoints to TwiBot NM and bot classification. | Archived; no new validation |
| `twibot20_transfer_out` | TwiBot-trained NM model to other graphs and classification tasks. | Archived; no new validation |

Exact recovery form (replace the family name):

```bash
git show archive/retired-analyses-2026-07-26:scripts/experiments/analysis/archive/<family>/<file>
```

### 9.2 Superseded records retained locally

| Record | Why retained | What supersedes it |
|---|---|---|
| [Objective rotation/pairs archive](archive/multitask_ssl_superseded/) | Historical MIX synergy claim and original node-task analyses | Valid LP rescore; no MIX-only LP capability |
| [MIX LP ablation](archive/mix_slp_ablation/FINDINGS.md) | Tests motivated by the invalid evaluator result | A causal test must target a capability established under valid scoring |
| [Old setup/evaluation exports](archive/outputs_old/) | Workflow and early task-construction history | Current protocol, catalog and per-experiment setup records |
| [Original full-graph NM audit](evaluation/error_audit/FINDINGS_NM_SOURCE_PAIR.md) and [bio clusters](evaluation/error_audit/FINDINGS_NM_BIO_CLUSTERS.md) | Explain how the initial HK interpretation arose | Corrected canonical-split audit; historical node-weighting reversal does not survive |

## 10. Coverage and maintenance

This snapshot enumerates **104 top-level setup families**, **74 analysis folders with Markdown/notebooks**, and **17 findings pages absent from main but present in other registered local worktrees**. These are inventory counts, not counts of independent experiments or successful runs. The git-only archive and related sibling-project families are listed separately.

The hierarchy groups multiple configs, budgets, seeds, replays and diagnostics under
the research question they address. Setup-only entries stay visible rather than being
silently counted as completed. Notebooks and exported tables may hold additional
results that need interpretation; presence alone does not supply a trustworthy verdict.

To update this overview, add the experiment to its research-question table with a
result/status and evidence link, retain the exact protocol and seed scope, and update
the relevant inventory entry. When a result is invalidated, change the interpretation
here and point to its replacement; do not merely append a warning beneath a false
headline. Move branch-only links to canonical relative paths when those results merge.

Known coverage limits: no live Tucker inventory was requested or performed; remote-only
runs, unregistered checkouts and results not exported into these local records may be
missing. Paper planning outside this repository is not counted as completed experimental
evidence. Side-branch conclusions are attributed to their records, not merged or rerun.
