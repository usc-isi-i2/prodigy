# Social-to-FB15K-237 compatibility transfer

## Question

Can the four nominated single-source social PRODIGY checkpoints solve the
original FB15K-237 few-shot relation-classification task, and does preserving
KG head/tail indicators improve that transfer?

## Protocol

- Downstream dataset: original public FB15K-237 artifact.
- Task: 20-way relation classification, 3 support examples and 4 queries per
  relation, 500 episodes (40,000 query predictions).
- Social checkpoints: COVID, TwiBot-20, Ukraine/Russia, and Midterm; training
  seed 0, checkpoint 2,500.
- Social architecture is kept fixed at `S,U,M`, 768 input dimensions, with no
  KG edge-feature module and no checkpoint fine-tuning.
- `text_768` supplies all 768 entity-text dimensions and omits head/tail flags.
- `text_766_endpoint_2` drops the last two text coordinates and replaces them
  with binary head/tail indicators, preserving the checkpoint's 768-wide input.
- `--ignore_label_embeddings True` is used in both conditions.

The evaluation episodes are the same in both conditions. In this codebase KG
evaluation episode sampling is determined by the split name, not `--seed`, so
these rows are a paired adapter comparison rather than an evaluation-seed
variance estimate.

Exact machine-readable values and run identifiers are in
[`data/results.csv`](data/results.csv). The source JSON files remain under the
corresponding Tucker run directories beneath
`/dataMeR1/phil/gfm/prodigy-social-fb15k/log/`.

## Results

| Social source | 768 text accuracy | 766 text + flags accuracy | Paired change |
|---|---:|---:|---:|
| COVID | 51.2600% | **51.3425%** | +0.0825 pp |
| TwiBot-20 | **41.6400%** | 37.6425% | -3.9975 pp |
| Ukraine/Russia | **41.3375%** | 41.3050% | -0.0325 pp |
| Midterm | **36.0525%** | 35.0150% | -1.0375 pp |
| Macro mean | **42.5725%** | 41.3263% | -1.2463 pp |

Ranking under the less invasive 768-text adapter is COVID, TwiBot-20,
Ukraine/Russia, then Midterm. All results are well above the 5% random accuracy
of a balanced 20-way episode, so the social checkpoints carry transferable
signal into a non-social knowledge graph. COVID is the strongest of the four at
51.26%.

Replacing text coordinates with endpoint flags does not provide a general
repair. It is effectively neutral for COVID and Ukraine/Russia, mildly harmful
for Midterm, and costs TwiBot-20 4.00 points. The macro mean falls 1.25 points.
This is consistent with the social checkpoints treating coordinates 766 and
767 as ordinary semantic features rather than as learned KG-role channels.

Two overlapping executions of the endpoint condition completed for COVID and
Ukraine/Russia during the queue-to-immediate-launch transition. Their accuracy
and F1 values reproduced exactly; the canonical rows in `results.csv` use the
later run identifiers. ROC-AUC differed only at floating-point noise scale.

## Comparison boundary

These are valid same-task transfer measurements for frozen social checkpoints,
but they are not strict reproductions of the paper's KG model. The native
Wiki-to-FB15K-237 route uses 770-dimensional inputs (768 text plus two endpoint
flags), KG edge features, and the deeper `S2,UX,M2` configuration. Our original-
style public reproduction reaches 73.795% accuracy at the nominated 8,001-update
checkpoint; the paper reports 72.04%. That gap cannot be attributed to source
pretraining alone because architecture and input interfaces differ as well.

## Decision

Use `text_768` as the primary frozen-checkpoint compatibility result. Retain
`text_766_endpoint_2` as a negative ablation, not as an improved adapter. A
source-controlled comparison against the paper requires retraining social
models with the native 770-dimensional, edge-aware `S2,UX,M2` KG-compatible
interface.
