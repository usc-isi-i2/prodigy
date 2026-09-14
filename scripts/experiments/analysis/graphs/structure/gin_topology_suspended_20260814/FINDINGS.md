# Ukraine-Suspended topology-only GIN sanity check

## Evidence status

**Admitted as historical, protocol-incomparable evidence.** The producing repository
committed this result on 2026-08-14, but the evaluated graph had 72,295 nodes while
the current canonical `ukr_rus_suspended` artifact has 56,440 nodes. The edge count
is the same (354,209). The result therefore describes the earlier padded-row graph
and must not be presented as an evaluation of the repaired September artifact.

The committed producing record remains available at revision
`3f82e42c3357c17283b922757883b48f0375a99a` in
`philippnoah/gin-topology`. Its referenced raw `results.json` was absent from the
recorded Tucker path when checked on 2026-09-14, so the exact aggregate cannot be
recomputed from raw outputs. The machine-readable admission receipt is in
[`data/validation_receipt.json`](data/validation_receipt.json).

## Historical result

All arms used the earlier 72,295-node, 354,209-edge Ukraine-Suspended graph. The
input was symmetrized to 676,968 unweighted edges and every node received the same
scalar feature. The balanced labeled-node split contained 1,000 train, 500
validation, and 1,000 test nodes, with recorded SHA-256 fingerprint
`7bab95b41c0c456dd3cca3fd37b4c0ae76a1a3173513b672058f3b409ea63082`.

| Arm | Test ROC-AUC | Test accuracy | Selected epochs |
|---|---:|---:|---:|
| Majority | 0.5000 | 0.5000 | — |
| Log-degree logistic regression | 0.5764 | 0.5650 | — |
| GIN, constant features | 0.5707 ± 0.0038 | 0.5400 ± 0.0370 | 5, 5, 22 |
| GIN, shuffled training labels | 0.5325 ± 0.0630 | 0.4933 ± 0.0133 | 1, 6, 18 |
| GIN, degree-preserving rewiring | 0.5689 ± 0.0158 | 0.5470 ± 0.0225 | 20, 6, 9 |

The values after `±` are sample standard deviations over seeds 0, 1, and 2.

## Supported interpretation

On the historical graph, the topology-only GIN was above chance but did not beat
the one-feature log-degree baseline. Degree-preserving rewiring left its mean AUC
nearly unchanged. This is evidence that the detectable topology-associated signal
was largely degree-level; it is not evidence that learned GIN structure improved
over degree, and it does not establish the same conclusion on the repaired graph.

## Exclusions

- Uncommitted TwiBot extension code in the retired Tucker checkout has no associated
  result record and is not admitted.
- The missing raw aggregate prevents replay or an independent metric recomputation.
- This node-classification sanity check is not directly comparable to the static-LP
  or episodic transfer evaluations elsewhere in PRODIGY.

## Provenance

- Producing repository: `https://github.com/philippnoah/gin-topology.git`
- Findings commit: `3f82e42c3357c17283b922757883b48f0375a99a`
- Recorded training-code revision: `8893072`
- Retired Tucker checkout inspected:
  `/dataMeR1/phil/gfm/experiment_archives/top_level_retired_20260912/gin-topology`
- Original recorded raw path, missing on 2026-09-14:
  `/dataMeR1/phil/gfm/gin-topology/runs/ukraine_suspended/results.json`
