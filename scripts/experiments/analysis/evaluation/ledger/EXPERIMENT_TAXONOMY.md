# Experiment taxonomy

Working taxonomy reconstructed from the repository setup tree, analysis index,
and W&B run names. Each experiment should receive one primary family and may
also receive secondary tags for architecture, objective, graph, task, or protocol.

## 1. Single-source transfer matrices

Experiments whose main object is a single-source train-graph by evaluation-graph grid.
Each model is trained on one source graph; merged training regimes do not define the
matrix family.

- Single-source transfer
- Single-source downstream task-transfer matrices
- Single-source cross-architecture comparisons
- Single-source native-objective model comparisons
- Single-source identity-disjoint and split-integrity controls

## 2. Transfer ladders

Experiments whose main object is an ordered progression of training sources or mixtures.

- PRODIGY source-addition ladders
- Graph-order and order-robustness ladders
- Facebook/source-set extensions
- Downstream transfer ladders
- SAMGPT / GraphCL native-objective ladders
- Weak-to-strong mixture ladders

## 3. Controlled ablations

Experiments that hold a main comparison fixed while changing one mechanism or protocol.

- Context depth and hop count
- Sampling schedule, strata, and cross-source shortcuts
- Merged-versus-single-source and balanced-versus-proportional regimes
- Batch construction and source exposure
- Encoder and architecture variants
- Feature/topology and feature-label ablations
- Train/test edge separation and identity-disjoint controls
- Center sampling and radius controls
- Correctness and evaluation-protocol ablations

## 4. Saturation and trajectories

Experiments that vary budget, training progress, or context scale.

- Pretraining-step saturation
- Data/source-exposure saturation
- Compute-matched context or hop saturation
- Radius/context-budget sweeps
- Checkpoint trajectories and convergence curves
- SAMGPT / GraphCL saturation studies

## 5. Objectives and pretext tasks

Experiments whose primary question is what the training objective learns.

- Neighbor matching
- Contrastive learning
- Masked feature prediction
- Multitask objective combinations
- Corpus-composition replications
- Rotation and paired-objective studies
- Topology-versus-feature capability
- Frozen-probe and pretraining-strategy comparisons

## 6. Graph and data diagnostics

Experiments that characterize graphs, features, overlap, or transfer predictors rather than comparing training procedures directly.

- Graph structural divergence
- Dataset identity overlap
- Biography-feature geometry
- Structure-feature/path coupling
- Similarity as a transfer predictor
- Graph construction and citation/tag smoke tests

## 7. Methods and validation

Infrastructure and correctness studies that support the experiments but are not themselves a model comparison family.

- Node-classification benchmark tables
- Node-regression benchmark tables
- Static-link-prediction benchmark tables
- Evaluator repairs
- Error and prediction audits
- Sampling/protocol validation
- Baseline, floor, and leakage checks

## 8. Replication, legacy, and setup-only work

Experiments that do not fit the main scientific families or are retained mainly for historical completeness.

- Paper and three-seed replications
- Legacy training batches
- Dataset-only or transfer-in setup work
- Historical cross-dataset evaluations
- Incomplete, superseded, or retired experiments

## Orthogonal tags and matrix axes

These are not primary experiment families. They describe the dimensions of a matrix, ladder, ablation, or saturation study.

- Architecture/model: PRODIGY, SAMGPT, VISION, GILT, raw features, supervised baselines
- Objective/regime: NM, contrastive, masked prediction, multitask, native GraphCL, supervised
- Source mixture: single graph, merged graphs, ordered ladder rung, corpus composition
- Checkpoint/budget: training step, data exposure, radius, hop count, compute budget
- Evaluation graph: target graph or held-out graph
- Downstream task: classification, regression, static link prediction, task transfer
- Evaluation protocol: shots, split, n-way, query count, feature/label setting, leakage control
- Replication identity: seed, order, run, checkpoint artifact, W&B run ID
