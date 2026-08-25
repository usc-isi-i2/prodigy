# Frozen-encoder adaptation efficiency

This experiment compares frozen learned encoders with the same linear-head
protocol. Downstream `CLS` means real node-label classification, never neighbor
matching.

## Registered protocol

- Label budgets: 0, 1, 10, and 100 labeled train-split nodes per class.
- Head updates: 0, 1, 10, and 100 full-batch AdamW updates at learning rate 0.01.
- The zero-label cell is evaluated only at update 0; no optimizer is constructed
  from or stepped on labels for that cell.
- Label-sampling seeds: 0, 1, and 2. Samples are balanced and nested within a
  seed, so the one-example sample is contained in the ten-example sample, which
  is contained in the hundred-example sample.
- Data split: the exact deterministic stratified 60/20/20 split used by the
  final-core SAMGPT downstream evaluation (`split_seed=0`).
- All learned representations are standardized from unlabeled train-split rows
  and mapped deterministically to 256 dimensions by truncation or zero padding.
  This makes the initialized 256-to-class linear head bit-identical across
  learned models for a target and label seed.
- Every result row records hashes of the split, sampled labeled nodes, and
  initialized head. The analysis rejects a cross-model mismatch rather than
  merely assuming those controls are shared.
- Metrics are recorded on the unchanged validation and test node sets at every
  requested update: ROC-AUC (macro one-vs-rest for multiclass) is primary, plus
  accuracy and macro-F1.
- Raw-feature logistic regression and a one-hidden-layer raw-feature MLP use the
  same data splits, label samples, update milestones, optimizer, and seeds. Their
  input dimensions differ from learned encoders, so initialization is matched by
  seed and PyTorch scheme rather than by an identical weight tensor.
- Graph encoders use each target artifact's canonical `graph.edge_index` for
  downstream classification. This is the same edge tensor used by the SAMGPT
  extractor and avoids depending on optional task-specific link-prediction views
  such as `static_train`, which are absent from several classification graphs.

`protocol.py` owns the contract. `run_head_grid.py` consumes checksumable feature
caches generated from each frozen encoder. The final analysis must preserve all
cells, compute label-efficiency area under ROC-AUC versus log10(label budget),
and report the first update reaching 95% of each budget-specific update-100
performance.

## Representation boundary

PRODIGY uses its pooled subgraph embedding before metagraph label propagation;
SAMGPT uses its frozen base-GCN embedding; GraphSAGE uses the frozen pilot-v1
node-history encoder. PRODIGY, SAMGPT, and GraphSAGE receive the same canonical
target edge tensor; architecture-native direction handling is retained. VISION
does not expose a support-independent whole-graph
embedding because its dual-view blocks globally attend to support nodes. The
registered label-free VISION probe therefore uses the checkpoint's learned
`feature_encoder` output. This is a conservative, explicitly topology-free view
of the learned VISION checkpoint and must not be described as the full
label-injected VISION predictor.
