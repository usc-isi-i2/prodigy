# Frozen encoder/decoder layer trace

This is a diagnostic of the admitted fresh-negative joint model, seed 0 at its
standalone-selected update 150. It does not train or select a model. The script
loads the exact checkpoint, pre-2018 graph, 2018 validation panel,
standardization, and saved score archive. Producer-GPU replay is exact. A local
CPU fallback must stay within 3e-6 absolute logit error and exactly preserve the
positive hitmask, negative top-50 membership, and official Hits@50.

The three SAGE outputs are captured after each convolution (ReLU after layers 1
and 2, as in production). Each 256-dimensional output is passed through the
unchanged frozen decoder with the unchanged standardized 14 structural inputs.
Layers 1 and 2 were not the representation on which the decoder was trained, so
their metrics are off-distribution interventions, not alternative models.

Two final-layer decoder-input ablations zero either all 512 learned pair inputs
(product and absolute difference) or all 14 structural inputs. Zero is also
off-distribution and does not retrain the decoder. These ablations locate
dependence in the frozen computation; they do not estimate the performance of a
properly trained structure-only or embedding-only model.

For raw node features and every SAGE layer, report endpoint-cosine AUC and
Hits@50, activation norms, and positive/negative distributions. For decoder
scores, report official full-panel Hits@50, cutoff, AUC, recovery/loss relative
to full scoring, and novel/repeat and zero/nonzero-AA strata. Separately report
the already frozen sample of 40 misses; it is interpretation only and never a
fitted subset.

No labels enter an optimizer, calibration, probe, checkpoint choice, feature
construction, or intervention choice. No 2019 data are read or scored. The 2018
panel is repeatedly explored development data, and all prior test exploration
and invalid leakage/test-supervised campaigns remain disclosed.

Run on Tucker from an isolated worktree using GPU 1. Runtime output must include
checkpoint/input hashes, producing revision, replay assertions, official
evaluator parity, tensor shapes, and a machine-readable result.
